

from dataclasses import dataclass
from collections import defaultdict
import logging, random
import numpy as np
import torch
import math

from texasholdem.game.game import TexasHoldEm
from texasholdem.game.action_type import ActionType

# ------------------------------- Буферы --------------------------------------

@dataclass
class Sample:
    state_features: dict         
    allowed_actions: list         
    advantage_targets: list       
    target_policy: list           
    is_weight: float = 1.0        

    priority: float = 1.0         
    per_is_weight: float = 1.0    

class StageBuffers:
    """
     стратифицированный буфер 
    """
    def __init__(self, per_stage_limit: dict | None = None):
        self.by_stage = defaultdict(list)
        self.seen_per_stage = defaultdict(int)
        self.per_stage_limit = per_stage_limit or {}
        self.total_seen = 0

    def add(self, sample: "Sample", phase: int, global_limit: int | None = None):
        """
        Новый образец фазы просто вытесняет старый:
        """
        if global_limit is not None:
            total = sum(len(b) for b in self.by_stage.values())
            if total >= int(global_limit):
                self.total_seen += 1
                self.seen_per_stage[int(phase)] += 1
                return False
        ph = int(phase)
        self.total_seen += 1
        self.seen_per_stage[ph] += 1
        bucket = self.by_stage[ph]
        k = self.per_stage_limit.get(ph)
        if k is None or len(bucket) < k:
            bucket.append(sample)
            return True
        seen = self.seen_per_stage[ph]
        j = random.randint(0, seen - 1)
        if j < k:
            bucket[j] = sample
            return True
        return False

    def sample_batch(self, batch_size: int, stage_mix: dict[int, float] | None = None, alpha: float = 0.6, beta: float = 0.4):
        stages = list(self.by_stage.keys())
        if not stages:
            return [], [], []
        if stage_mix is None:
            stage_mix = {ph: 1.0 / len(stages) for ph in stages}
        out_samples, out_indices, out_phases = [], [], []
        for ph in stages:
            bucket = self.by_stage[ph]
            if not bucket:
                continue
            n_take = max(1, int(round(batch_size * stage_mix.get(ph, 0.0))))
            pri = torch.tensor([max(1e-6, s.priority) for s in bucket], dtype=torch.float32)
            probs = pri.pow(alpha)
            probs /= probs.sum()
            idxs = torch.multinomial(probs, num_samples=min(n_take, len(bucket)), replacement=len(bucket) < n_take).tolist()
            N = len(bucket)
            p_sel = probs[idxs]
            per_w = (N * p_sel).pow(-beta)
            per_w = per_w / per_w.mean().clamp_min(1e-6)
            for j, w in zip(idxs, per_w.tolist()):
                bucket[j].per_is_weight = float(w)
            out_samples.extend([bucket[j] for j in idxs])
            out_indices.extend(idxs)
            out_phases.extend([ph] * len(idxs))
        return out_samples, out_indices, out_phases

    def update_priorities(self, phases: list[int], indices: list[int], new_priorities: list[float]):
        """
        обновляет приоритеты после шага обучения 
        """
        for ph, idx, pr in zip(phases, indices, new_priorities):
            b = self.by_stage[int(ph)]
            if 0 <= idx < len(b):
                b[idx].priority = float(max(1e-6, pr))

    def merged(self):
        """
        Экспорт всех сэмплов
        """
        merged = []
        for ph, lst in self.by_stage.items():
            merged.extend(lst)
        return merged, self.by_stage



# Дерево подигры выдающее разведанные игровые узлы

class GameTreeSampler:
    """
    Ветвление происходит только в узлах избранного (случайно) игрока
    Для оппонента/шанса — external sampling по стратегии сети + epsilon.
    Из-за большого пространства возможных действий в безлимитной игре
    используется MC-CFR, предполагающий ветвления по ограниченному 
    количеству действий M(гиперпараметр)
    """
    def __init__(self, simulator, stage_buffers=None, global_sample_limit: int = 1_000_000, M: int = 2):
        self.sim = simulator
        self.stage_buffers = stage_buffers or StageBuffers()
        self.global_sample_limit = int(global_sample_limit)
        self.M = int(M)
        self.behavior_mode = getattr(self.sim, 'behaviro_mode', 'mixture')
        self.behavior_alpha = getattr(self.sim, 'behaviro_alpha', 0.05)
        

    def _policy_and_adv(self, POS_list):
        device = next(self.sim.model.parameters()).device
        with torch.no_grad():
            policy_probs, adv_logits = self.sim.model.forward(**self.sim.pack_to_tensor(POS_list))
        policy_np = policy_probs.squeeze(0).detach().cpu().numpy()
        adv_np = adv_logits.squeeze(0).detach().cpu().numpy()
        mask_np = np.array(POS_list['allowed_actions'], dtype=bool)
        adv_np[~mask_np] = 0.0
        return policy_np, adv_np

    def _sample_opponent_action(self, game, player_id):
        """
        Выбор действия оппонента,
        """
        POS = self.sim.private_observation_space(player_id, game)
        mask_np = np.array(POS['allowed_actions'], dtype=bool)
        if mask_np.sum() == 0:
            raise RuntimeError("неверное состояние" \
            "_sample_opponent_action Error")
        with torch.no_grad():
            policy_probs, _ = self.sim.model.forward(**self.sim.pack_to_tensor(POS))
        pi = policy_probs.squeeze(0).detach().cpu().numpy()
        pi[~mask_np] = 0.0
        s = pi.sum()
        pi = pi / s if mask_np.sum() > 0 and s > 0 else pi
        if self.behavior_mode == 'mixture':
            q = (1.0 - self.behavior_alpha) * pi + self.behavior_alpha * (mask_np.astype(float) / mask_np.sum())
            s = q.sum()
            if not np.isfinite(s) or s <= 0:
                allowed = np.flatnonzero(mask_np)
                q = np.zeros_like(pi)
                q[allowed] = 1.0 / len(allowed)
            else:
                q = q / s
            a_idx = int(np.random.choice(len(q), p=q))
            w_node = (pi[a_idx] / max(q[a_idx], 1e-12))
            return a_idx, w_node
        elif self.behavior_mode == 'topk_full_target':
            k = getattr(self.sim, 'behaviro_topk', 4)
            allowed = np.flatnonzero(mask_np)
            if allowed.size == 0:
                raise RuntimeError("Нет доступных действйи" \
                "_sample_opponent_action Error")
            pi_idx_sorted = np.argsort(pi)[::-1]
            topk_idx = [i for i in pi_idx_sorted if i in allowed][:k]
            q = np.zeros_like(pi)
            if topk_idx:
                total_topk = pi[topk_idx].sum()
                q[topk_idx] = pi[topk_idx] / (total_topk if total_topk > 0 else 1.0)
            s = q.sum()
            if not np.isfinite(s) or s <= 0:
                q = np.zeros_like(pi)
                q[allowed] = 1.0 / allowed.size
                Zk = 1.0
            else:
                q = q / s
                Zk = float(pi[topk_idx].sum()) if topk_idx else 1.0
            a_idx = int(np.random.choice(len(q), p=q))
            w_node = Zk
            return a_idx, w_node
        elif self.behavior_mode == 'topk_truncated_target':
            k = getattr(self.sim, 'behaviro_topk', 4)
            allowed = np.flatnonzero(mask_np)
            q = np.zeros_like(pi)
            if allowed.size <= k:
                q[allowed] = 1.0 / allowed.size
            else:
                pi_idx_sorted = np.argsort(pi)[::-1]
                topk_idx = [i for i in pi_idx_sorted if i in allowed][:k]
                if topk_idx:
                    total_topk = pi[topk_idx].sum()
                    q[topk_idx] = pi[topk_idx] / (total_topk if total_topk > 0 else 1.0)
            s = q.sum()
            if not np.isfinite(s) or s <= 0:
                q = np.zeros_like(pi)
                q[allowed] = 1.0 / allowed.size
            else:
                q = q / s
            a_idx = int(np.random.choice(len(q), p=q))
            return a_idx, 1.0
        else:
            raise ValueError(f"неизвестный мод:={self.behavior_mode}")

    def traverse_history(self, action_history, traverser_id, seed, out_samples, W: float = 1.0):
        """
        External sampling с IS-весами
        На каждом узле traverser-игрока выбирается случайные M действий для углубления
        Регреты считаются только для выбранных действий остальные = 0
        MCCFR предполагает что при достаточном количестве итераций алгоритма - 
        модель рассмотрит все действия, как и те, что мы занулили
        """
        game = self._simulate_game_state(action_history, seed)
        if not game.is_hand_running():
            stacks = [p.chips for p in game.players]
            return stacks[traverser_id] - self.sim.buyin
        current_player = game.current_player
        if current_player != traverser_id:
            a_idx, w_node = self._sample_opponent_action(game, current_player)
            return self.traverse_history(action_history + [a_idx], traverser_id, seed, out_samples, W=W * float(w_node))
        POS = self.sim.private_observation_space(current_player, game)
        mask_np = np.array(POS['allowed_actions'], dtype=bool)
        phase = POS['phase']
        if mask_np.sum() == 0:
            return 0.0
        total_now = sum(len(b) for b in self.stage_buffers.by_stage.values())
        if self.global_sample_limit is not None and total_now >= int(self.global_sample_limit):
            return 0.0
        policy_np, adv_np = self._policy_and_adv(POS)
        allowed_actions = np.flatnonzero(mask_np)
        if allowed_actions.size == 0:
            return 0.0
        M = min(self.M, allowed_actions.size)
        chosen = list(np.random.choice(allowed_actions, size=M, replace=False))

        u_exact = np.zeros_like(policy_np, dtype=np.float32)
        for a_idx in chosen:
            u_exact[a_idx] = self.traverse_history(action_history + [a_idx], traverser_id, seed, out_samples, W=W)

        u_sel = [u_exact[a] for a in chosen]
        baseline = float(sum(u_sel) / max(1, len(u_sel)))

        regrets = np.zeros_like(policy_np, dtype=np.float32)
        for a_idx in chosen:
            regrets[a_idx] = float(W) * (u_exact[a_idx] - baseline)

        r_plus = np.clip(regrets, 0.0, None)
        total_pos = r_plus[mask_np].sum()
        target_policy = np.zeros_like(policy_np, dtype=np.float32)
        if total_pos > 1e-12:
            target_policy[mask_np] = r_plus[mask_np] / total_pos
        else:
            target_policy[mask_np] = 1.0 / mask_np.sum()


        total_pos = r_plus[mask_np].sum()
        target_policy = np.zeros_like(policy_np)
        if total_pos > 1e-12:
            target_policy[mask_np] = r_plus[mask_np] / total_pos
        else:
            target_policy[mask_np] = 1.0 / mask_np.sum()
        state_feat = POS.copy()
        allowed_mask = state_feat.pop('allowed_actions')
        sample = Sample(state_features=state_feat, allowed_actions=allowed_mask, advantage_targets=r_plus.tolist(), target_policy=target_policy.tolist(), is_weight=float(W))
        self.stage_buffers.add(sample, phase=phase, global_limit=self.global_sample_limit)
        return 0.0

    def _simulate_game_state(self, action_history, seed):
        """
        Пересоздаём игру до выбранного действия включительно, чтобы 
        продолжить рекурсию с фиксированными random_seed
        """
        random.seed(seed)
        game = TexasHoldEm(buyin=self.sim.buyin, big_blind=self.sim.big_blind, small_blind=self.sim.small_blind, max_players=self.sim.num_players)
        game.start_hand()
        for a_idx in action_history:
            a_type, a_param = self.sim.get_action_from_index(a_idx, game=game)
            if isinstance(a_param, dict):
                game.take_action(a_type, **a_param)
            elif a_param is None:
                game.take_action(a_type)
            else:
                game.take_action(a_type, a_param)
        return game

    def generate_samples(self, traverser_id, fixed_seed=None):
        """
        Генерирует набор обучающих примеров для одного запуска игры
        """
        self.stage_buffers = StageBuffers()

        seed = fixed_seed if fixed_seed is not None else random.randrange(1, 2**32)

        logging.info(f"Traversal start: traverser={traverser_id}, seed={seed}")

        self.traverse_history([], traverser_id, seed, [], W=1.0)
        merged_samples = []
        for phase, lst in self.stage_buffers.by_stage.items():
            merged_samples.extend(lst)
        return merged_samples, self.stage_buffers.by_stage
