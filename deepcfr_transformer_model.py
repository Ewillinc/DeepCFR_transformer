import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_softmax(logits: torch.Tensor, legal_mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Мы ориентируемся на архитектуру texasholdem где одно действие fold всегда доступно
    но, наша уверенность не 100%
    Полностью пустая маска может сыграть в убытк.
    """
    neg_inf = torch.finfo(logits.dtype).min
    masked = logits.masked_fill(~legal_mask, neg_inf)
    out = torch.softmax(masked, dim=dim)

    sums = out.sum(dim=dim, keepdim=True)
    bad = torch.isnan(out) | (sums == 0)

    if bad.any():
        legal_counts = legal_mask.sum(dim=dim, keepdim=True).clamp_min(1)
        uniform_legal = legal_mask.float() / legal_counts
        out = torch.where(bad, uniform_legal, out)
    return out



class DeepCFR(nn.Module):
    def __init__(self, num_players=8, card_emb_dim=32, player_emb_dim=16,
                 hidden_dim=128, n_attention_heads=4, n_attention_layers=2,
                 num_actions=10, buyin: int = 20000):
        
        
        super(DeepCFR, self).__init__()
        self.num_players = num_players
        self.card_emb_dim = card_emb_dim
        self.player_emb_dim = player_emb_dim
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions
        drop_p = 0.10

        self.buyin = buyin
        self.card_embedding = nn.Embedding(num_embeddings=53, embedding_dim=card_emb_dim, padding_idx= 52)
        self.player_embedding = nn.Embedding(num_embeddings=num_players, embedding_dim=player_emb_dim)
        self.phase_embedding = nn.Embedding(num_embeddings=6, embedding_dim=8)

        self.player_info_linear = nn.Linear(2, player_emb_dim)
        self.pot_linear = nn.Linear(3, 8)                                         

        self.money_ln       = nn.LayerNorm(3, eps = 1e-5)   
        self.player_num_ln  = nn.LayerNorm(2, eps = 1e-5)   

        self.private_linear = nn.Linear(2 * card_emb_dim, hidden_dim)
        self.action_mask_linear = nn.Linear(num_actions, hidden_dim)
        self.global_proj = nn.Linear(16, hidden_dim)

        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=n_attention_heads,
                                  dropout = drop_p, batch_first=True)
            for _ in range(n_attention_layers)
        ])
        self.attn_layernorms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_attention_layers)])
        self.attn_ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(drop_p),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Dropout(drop_p),
            )
            for _ in range(n_attention_layers)
        ])

        self.ln_global  = nn.LayerNorm(hidden_dim)  
        self.ln_mask    = nn.LayerNorm(hidden_dim)  
        self.ln_board   = nn.LayerNorm(hidden_dim)  
        self.ln_player  = nn.LayerNorm(hidden_dim)  
        self.ln_priv    = nn.LayerNorm(hidden_dim)  

        self.logit_gain = nn.Parameter(torch.tensor(0.0))   
        self.adv_gain = nn.Parameter(torch.tensor(0.0))     
        self._min_scale = 0.5                               
        self._max_scale = 10.0                              

        self.attn_ffn_layernorms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_attention_layers)])
        self.policy_head = nn.Linear(hidden_dim, num_actions)
        self.advantage_head = nn.Linear(hidden_dim, num_actions) 

        self.d_aux = 8  
        self.type_embedding = nn.Embedding(5, self.d_aux) 
        self.seat_rel_actor_emb = nn.Embedding(self.num_players, self.d_aux)                    
        self.board_pos_emb = nn.Embedding(5, self.d_aux)                                        
        self.aux2hid = nn.Linear(self.d_aux, self.hidden_dim)                                   

    def forward(self,
                phase,            
                pot,              
                community_cards,       
                player_chips,          
                player_bets,           
                player_active,         
                current_player,   
                private_cards,         
                allowed_actions,       
                to_call,          
                call_is_allin,    
                seat_rel_actor,         
                board_count            
                ):
        
        device = self.card_embedding.weight.device
        B, A = allowed_actions.shape
        N = player_chips.shape[1]

        def add_aux(x, *aux_list):
            if not aux_list:
                return x
            s = None
            for a in aux_list:
                if a is None:
                    continue
                s = a if s is None else s + a
            if s is None:
                return x
            s = self.aux2hid(s)
            return x + s
        

        phase_emb = self.phase_embedding(phase)                                
        money_feats = torch.stack([pot, to_call, call_is_allin], dim=-1)        
        money_feats = self.money_ln(money_feats)                                                    
        money_emb = self.pot_linear(money_feats)                                
        global_tok = self.global_proj(torch.cat([phase_emb, money_emb], dim=-1))

        TYPE_GLOBAL, TYPE_MASK, TYPE_BOARD, TYPE_PLAYER, TYPE_PRIVATE = 0, 1, 2, 3, 4
        E_t_global = self.type_embedding(torch.full((B,), TYPE_GLOBAL, dtype=torch.long, device=device))  
        global_tok = add_aux(global_tok, E_t_global)   

        global_tok = self.ln_global(global_tok)                                                     

        mask_tok = self.ln_mask(self.action_mask_linear(allowed_actions))                          
       
        E_t_mask = self.type_embedding(torch.full((B,), TYPE_MASK, dtype=torch.long, device=device))       
        mask_tok = add_aux(mask_tok, E_t_mask)         

        comm_embs = self.card_embedding(community_cards)                        
        if self.card_emb_dim < self.hidden_dim:
            pad = torch.zeros(B, 5, self.hidden_dim - self.card_emb_dim, device=device)
            comm_toks = torch.cat([comm_embs, pad], dim=-1)                     
        else:
            comm_toks = comm_embs[..., :self.hidden_dim]                        


        E_t_board  = self.type_embedding(torch.full((B, 5), TYPE_BOARD, dtype=torch.long, device=device))  
        board_idx  = torch.arange(5, dtype=torch.long, device=device).unsqueeze(0).expand(B, -1)           
        E_boardpos = self.board_pos_emb(board_idx)                                                         
        comm_toks  = self.ln_board(add_aux(comm_toks, E_t_board, E_boardpos))  

        pid_range = torch.arange(self.num_players, device=device)          
        pid_embs  = self.player_embedding(pid_range).unsqueeze(0)          
        pid_embs  = pid_embs.expand(B, -1, -1)                             

        num_feats = torch.stack([player_chips, player_bets], dim=-1)       
        num_feats = self.player_num_ln(num_feats)                                                   
        num_embs  = self.player_info_linear(num_feats) 

        active = player_active.unsqueeze(-1).float()                      
        pid_embs  = pid_embs  * active                                     
        num_embs  = num_embs  * active                                     

        player_vec = torch.cat([pid_embs, num_embs], dim=-1)               
        if player_vec.size(-1) < self.hidden_dim:
            pad = torch.zeros(B, self.num_players,
                            self.hidden_dim - player_vec.size(-1), device=device)
            player_toks = torch.cat([player_vec, pad], dim=-1)             
        else:
            player_toks = player_vec[..., :self.hidden_dim]  

        player_toks = self.ln_player(player_toks)                                                   
        
        E_t_player = self.type_embedding(torch.full((B, self.num_players), TYPE_PLAYER,
                                                   dtype=torch.long, device=device))                       
        E_seat     = self.seat_rel_actor_emb(seat_rel_actor)                                               
        player_toks = add_aux(player_toks, E_t_player, E_seat)

        priv_embs = self.card_embedding(private_cards).reshape(B, -1)           
        priv_tok  = self.private_linear(priv_embs)                              

        E_t_priv  = self.type_embedding(torch.full((B, 2), TYPE_PRIVATE, dtype=torch.long, device=device)) 
        priv_tok  = self.ln_priv(add_aux(priv_tok, E_t_priv[:,0,:])) 

        seq = torch.cat([
            global_tok.unsqueeze(1),   
            mask_tok.unsqueeze(1),     
            comm_toks,                 
            player_toks,               
            priv_tok.unsqueeze(1)      
        ], dim=1)                      

        T = seq.size(1)
        pad_mask = torch.zeros(B, T, dtype=torch.bool, device=device)  

        board_idx_mask = torch.arange(5, device=device).unsqueeze(0)              
        board_active   = (board_idx_mask < board_count.unsqueeze(1))              
        pad_mask[:, 2:7] = ~board_active    

        pad_mask[:, 7:7 + self.num_players] = (player_active == 0.0)       
        
        x = seq
        for attn, ln1, ffn, ln2 in zip(self.attention_layers,
                                       self.attn_layernorms,
                                       self.attn_ffns,
                                       self.attn_ffn_layernorms):
            x = ln1(x + attn(x, x, x, need_weights=False, key_padding_mask=pad_mask)[0])
            x = ln2(x + ffn(x))

        player_tokens_start = 1 + 1 + 5
        idx = player_tokens_start + current_player            
        batch_idx = torch.arange(B, device=device)
        cur_tok = x[batch_idx, idx]                           

        scale_pol = torch.clamp(self.logit_gain.exp(), self._min_scale, self._max_scale)
        scale_adv = torch.clamp(self.adv_gain.exp(), self._min_scale, self._max_scale)

        logits_pol = scale_pol * self.policy_head(cur_tok)                  
        legal_mask = allowed_actions.bool()
        policy_probs = masked_softmax(logits_pol, legal_mask) 

        adv_raw = scale_adv * self.advantage_head(cur_tok)                  
        adv = adv_raw.masked_fill(~legal_mask, 0.0)           

        return policy_probs, adv