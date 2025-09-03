from typing import List, Dict, Tuple, Any
import torch
import torch.nn as nn
import torch.optim as optim
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

def _to_tensors_from_samples(samples: List[Any], device: torch.device, num_players: int, num_actions: int):
    """
    Превращает список Sample в батч тензоров для DeepCFR.
    """
    phase, pot, comm, pchips, pbets, pact, cur, priv = [], [], [], [], [], [], [], []
    allowed, to_call, call_is_ai, seat_rel, board_cnt = [], [], [], [], []
    w_path_list, w_per_list = [], []

    for s in samples:
        sf = s.state_features
        if isinstance(sf, dict):
            phase.append(sf.get("phase", 0))
            pot.append(sf.get("pot", 0.0))
            comm.append(sf.get("community_cards", [52]*5))
            pchips.append(sf.get("player_chips", [0.0]*num_players))
            pbets.append(sf.get("player_bets", [0.0]*num_players))
            pact.append(sf.get("player_active", [0]*num_players))
            cur.append(sf.get("current_player", 0))
            priv.append(sf.get("private_cards", [52, 52]))
            to_call.append(sf.get("to_call", 0.0))
            call_is_ai.append(sf.get("call_is_allin", 0))
            seat_rel.append(sf.get("seat_rel_actor", list(range(num_players))))
            board_cnt.append(sf.get("board_count", 0))
        else:
            _phase, _pot, _comm, _pchips, _pbets, _pact, _cur, _priv = sf
            phase.append(_phase); pot.append(_pot); comm.append(_comm)
            pchips.append(_pchips); pbets.append(_pbets); pact.append(_pact)
            cur.append(_cur); priv.append(_priv)
            to_call.append(0.0); call_is_ai.append(0); seat_rel.append(list(range(num_players)))
            board_cnt.append(sum(int(c != 52) for c in _comm))

        allowed.append(s.allowed_actions)
        w_path_list.append(float(getattr(s, "is_weight", 1.0)))
        w_per_list.append(float(getattr(s, "per_is_weight", 1.0)))

    B = len(samples)
    t = lambda x, dt: torch.tensor(x, dtype=dt, device=device)

    return {
        'phase': t(phase, torch.long),
        'pot': t(pot, torch.float32),
        'community_cards': t(comm, torch.long).view(B, 5),
        'player_chips': t(pchips, torch.float32).view(B, num_players),
        'player_bets': t(pbets, torch.float32).view(B, num_players),
        'player_active': t(pact, torch.float32).view(B, num_players),
        'current_player': t(cur, torch.long),
        'private_cards': t(priv, torch.long).view(B, 2),
        'allowed_actions': t(allowed, torch.float32).view(B, num_actions),
        'to_call': t(to_call, torch.float32),
        'call_is_allin': t(call_is_ai, torch.long),
        'seat_rel_actor': t(seat_rel, torch.long).view(B, num_players),
        'board_count': t(board_cnt, torch.long),

        'weights': t(w_path_list, torch.float32),       
        'per_weights': t(w_per_list, torch.float32),    
    }

class DeepCFRTrainer:
    """
    Обучение DeepCFR_transfromer
      Обучает advantage_head по MSE(r_plus)
      Обучает policy_head по KL-дивергенции(target_policy , policy_pred)
      Итоговый вес лосса = is_weight * per_is_weight
      выбор батчей через StageBuffers.sample_batch + update_priorities
    """
    def __init__(self, simulator, sampler, batch_size: int = 512, adv_steps: int = 2000, pol_steps: int = 2000,
                 lr: float = 3e-4, weight_decay: float = 0.0, grad_clip: float | None = 1.0,
                 per_alpha: float = 0.6, per_beta: float = 0.4, tb_logdir: str | None = None):
        self.sim = simulator
        self.sampler = sampler
        self.model = self.sim.model
        self.device = self.sim.model_device()
        self.num_players = self.sim.num_players
        self.num_actions = self.sim.num_actions

        self.batch_size = batch_size
        self.adv_steps = adv_steps
        self.pol_steps = pol_steps
        self.grad_clip = grad_clip

        self.per_alpha = float(per_alpha)
        self.per_beta = float(per_beta)

        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.mse = nn.MSELoss(reduction='none')
        self.global_step = 0
        self.tb = SummaryWriter(tb_logdir) if (tb_logdir and SummaryWriter is not None) else None

    def train_on_buffer(self):
        adv_loss_avg = self._train_advantages()
        pol_loss_avg = self._train_policy()
        return adv_loss_avg, pol_loss_avg

    def _train_advantages(self):
        losses_adv = []
        for _ in range(self.adv_steps):
            batch, idxs, phases = self.sampler.stage_buffers.sample_batch(
                batch_size=self.batch_size, stage_mix=None, alpha=self.per_alpha, beta=self.per_beta
            )
            if not batch:
                break

            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)

            batch_t = _to_tensors_from_samples(batch, self.device, self.num_players, self.num_actions)
            inputs = {
                'phase': batch_t['phase'],
                'pot': batch_t['pot'],
                'community_cards': batch_t['community_cards'],
                'player_chips': batch_t['player_chips'],
                'player_bets': batch_t['player_bets'],
                'player_active': batch_t['player_active'],
                'current_player': batch_t['current_player'],
                'private_cards': batch_t['private_cards'],
                'allowed_actions': batch_t['allowed_actions'],
                'to_call': batch_t['to_call'],
                'call_is_allin': batch_t['call_is_allin'].float(),
                'seat_rel_actor': batch_t['seat_rel_actor'],
                'board_count': batch_t['board_count'],
            }
            target_adv = torch.tensor([s.advantage_targets for s in batch], dtype=torch.float32, device=self.device)

            pi_pred, adv_pred = self.model(**inputs)

            legal = (batch_t['allowed_actions'] > 0.5).float()
            legal_count = legal.sum(dim=1).clamp_min(1.0)

            adv_mse_full = self.mse(adv_pred, target_adv)
            adv_mse = (adv_mse_full * legal).sum(dim=1) #/ legal_count убрал деление, чтобы не урезать loss

            w_path = batch_t['weights']
            w_per = batch_t['per_weights']
            w_full = w_path * w_per
            w_sum = w_full.sum().clamp_min(1e-12)

            adv_loss = (w_full * adv_mse).sum() / w_sum
            adv_loss.backward()
            if self.grad_clip is not None:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            self.global_step += 1

            with torch.no_grad():
                pr = adv_mse.detach().cpu().numpy().tolist()
            self.sampler.stage_buffers.update_priorities(phases, idxs, pr)

            losses_adv.append(float(adv_loss.item()))
            if self.tb:
                self.tb.add_scalar("train/adv_loss", losses_adv[-1], self.global_step)

        return (sum(losses_adv[-100:]) / max(1, len(losses_adv[-100:]))) if losses_adv else 0.0

    def _train_policy(self):
        losses = []
        for _ in range(self.pol_steps):
            batch, idxs, phases = self.sampler.stage_buffers.sample_batch(
                batch_size=self.batch_size, stage_mix=None, alpha=self.per_alpha, beta=self.per_beta
            )
            if not batch:
                break

            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)

            batch_t = _to_tensors_from_samples(batch, self.device, self.num_players, self.num_actions)
            inputs = {
                'phase': batch_t['phase'],
                'pot': batch_t['pot'],
                'community_cards': batch_t['community_cards'],
                'player_chips': batch_t['player_chips'],
                'player_bets': batch_t['player_bets'],
                'player_active': batch_t['player_active'],
                'current_player': batch_t['current_player'],
                'private_cards': batch_t['private_cards'],
                'allowed_actions': batch_t['allowed_actions'],
                'to_call': batch_t['to_call'],
                'call_is_allin': batch_t['call_is_allin'].float(),
                'seat_rel_actor': batch_t['seat_rel_actor'],
                'board_count': batch_t['board_count'],
            }
            target_pi = torch.tensor([s.target_policy for s in batch], dtype=torch.float32, device=self.device)
            pi_pred, _ = self.model(**inputs)

            eps = 1e-8
            tgt = torch.clamp(target_pi, min=eps, max=1.0)
            pred = torch.clamp(pi_pred, min=eps, max=1.0)
            kl_per_sample = torch.sum(tgt * (torch.log(tgt) - torch.log(pred)), dim=1)

            # >>> теперь есть в batch_t
            w_path = batch_t['weights']
            w_per = batch_t['per_weights']
            w_full = w_path * w_per
            loss = (w_full * kl_per_sample).sum() / (w_full.sum().clamp_min(1e-12))

            loss.backward()
            if self.grad_clip is not None:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            self.global_step += 1

            with torch.no_grad():
                pr = kl_per_sample.detach().cpu().numpy().tolist()
            self.sampler.stage_buffers.update_priorities(phases, idxs, pr)

            losses.append(float(loss.item()))
            if self.tb:
                self.tb.add_scalar("train/policy_loss", losses[-1], self.global_step)

        return (sum(losses[-100:]) / max(1, len(losses[-100:]))) if losses else 0.0

    def state_dict(self) -> Dict[str, Any]:
        return {"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict()}

    def load_state_dict(self, state: Dict[str, Any]):
        self.model.load_state_dict(state.get("model", {}))
        self.optimizer.load_state_dict(state.get("optimizer", {}))

    def close(self):
        if self.tb:
            self.tb.close()
