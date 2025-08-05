
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepCFR(nn.Module):
    def __init__(self, num_players=8, card_emb_dim=32, player_emb_dim=16,
                 hidden_dim=128, n_attention_heads=4, n_attention_layers=2,
                 num_actions=10):
        

        super(DeepCFR, self).__init__()
        self.num_players = num_players
        self.card_emb_dim = card_emb_dim
        self.player_emb_dim = player_emb_dim
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions
        drop_p = 0.10



        self.card_embedding = nn.Embedding(num_embeddings=53, embedding_dim=card_emb_dim)
        self.player_embedding = nn.Embedding(num_embeddings=num_players, embedding_dim=player_emb_dim)
        self.phase_embedding = nn.Embedding(num_embeddings=6, embedding_dim=8)

        self.player_info_linear = nn.Linear(2, player_emb_dim)
        self.pot_linear = nn.Linear(1, 8)

        self.private_linear = nn.Linear(2 * card_emb_dim, hidden_dim)

        self.action_mask_linear = nn.Linear(num_actions, hidden_dim)

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
        self.attn_ffn_layernorms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_attention_layers)])
        self.policy_head = nn.Linear(hidden_dim, num_actions)
        self.advantage_head = nn.Linear(hidden_dim, num_actions) 
        

    def forward(
        self,
        phase: torch.Tensor,        
        pot: torch.Tensor,          
        community_cards: torch.Tensor,  
        player_chips: torch.Tensor,     
        player_bets: torch.Tensor,      
        player_active: torch.Tensor,    
        current_player: torch.Tensor,   
        private_cards: torch.Tensor,    
        allowed_actions: torch.Tensor,  
    ):

        device = self.card_embedding.weight.device
        B, A = allowed_actions.shape

        phase_emb = self.phase_embedding(phase)                        
        pot_emb   = self.pot_linear(pot.unsqueeze(-1))                 
        global_tok = torch.zeros(B, self.hidden_dim, device=device)
        global_tok[:, :8] = phase_emb + pot_emb                        

        # ---------- маска действий ----------
        mask_tok = self.action_mask_linear(allowed_actions)            

        # ---------- стол ----------
        comm_embs = self.card_embedding(community_cards)               
        if self.card_emb_dim < self.hidden_dim:                        
            pad = torch.zeros(B, 5, self.hidden_dim - self.card_emb_dim, device=device)
            comm_toks = torch.cat([comm_embs, pad], dim=-1)            
        else:
            comm_toks = comm_embs[..., :self.hidden_dim]

        # ---------- игроки ----------
        pid_range = torch.arange(self.num_players, device=device)      
        pid_embs  = self.player_embedding(pid_range)                  
        pid_embs  = pid_embs.unsqueeze(0).expand(B, -1, -1)           
        num_feats = torch.stack([player_chips, player_bets], dim=-1)  
        num_embs  = self.player_info_linear(num_feats)                
        num_embs += player_active.unsqueeze(-1).float()                
        player_vec = torch.cat([pid_embs, num_embs], dim=-1)           
        if player_vec.size(-1) < self.hidden_dim:
            pad = torch.zeros(B, self.num_players,
                               self.hidden_dim - player_vec.size(-1), device=device)
            player_toks = torch.cat([player_vec, pad], dim=-1)         
        else:
            player_toks = player_vec[..., :self.hidden_dim]

        # ---------- приватные карты ----------
        priv_embs = self.card_embedding(private_cards)                 
        priv_tok  = self.private_linear(priv_embs.view(B, -1))    

        seq = torch.cat([
            global_tok.unsqueeze(1),           
            mask_tok.unsqueeze(1),             
            comm_toks,                         
            player_toks,                       
            priv_tok.unsqueeze(1)              
        ], dim=1)                              

        x = seq
        for attn, ln1, ffn, ln2 in zip(self.attention_layers,
                                       self.attn_layernorms,
                                       self.attn_ffns,
                                       self.attn_ffn_layernorms):
            x = ln1(x + attn(x, x, x, need_weights=False)[0])          
            x = ln2(x + ffn(x))

        player_tokens_start = 1 + 1 + 5                                 
        idx = player_tokens_start + current_player                      
        batch_idx = torch.arange(B, device=device)
        cur_tok = x[batch_idx, idx]                                     

        # ---------- головы ----------

        logits_adv = self.advantage_head(cur_tok)                       
        logits_adv = logits_adv.masked_fill(~allowed_actions.bool(), float('-inf'))

        adv = logits_adv 

        logits_pol = self.policy_head(cur_tok)                         
        logits_pol = logits_pol.masked_fill(~allowed_actions.bool(), float('-inf'))
        policy_probs = torch.softmax(logits_pol, dim=-1)
        return  policy_probs, adv
