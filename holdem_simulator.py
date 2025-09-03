import numpy as np
import torch
import random
from texasholdem.game.game import TexasHoldEm
from texasholdem.game.action_type import ActionType
from texasholdem.game.hand_phase import HandPhase
from texasholdem.card.card import Card
from deepcfr_transformer_model import DeepCFR

class HoldemSimulator:
    def __init__(self, behaviro_mode='mixture', behaviro_alpha=0.05, behaviro_topk=4, clip_si_weight: float | None = None,
                 num_players=8, buyin=200, big_blind=1, small_blind=0.5, normalize_money=False, **model_params):

        self.behaviro_mode = behaviro_mode
        self.behaviro_alpha = behaviro_alpha
        self.behaviro_topk = behaviro_topk
        self.clip_is_weight = clip_si_weight
        self.num_players = num_players
        self.buyin = buyin
        self.big_blind = big_blind
        self.small_blind = small_blind
        self.normalize_money = normalize_money
        self.num_actions = 10

        self.game = TexasHoldEm(buyin=buyin, big_blind=big_blind, small_blind=small_blind, max_players=num_players)
        self.device = torch.device('cpu')
        self.model = DeepCFR(num_players=num_players, **model_params).to(self.device)
        self.card_dict = {
            Card('2c'): [0, 0], Card('2d'): [1, 0], Card('2h'): [2, 0], Card('2s'): [3, 0],
            Card('3c'): [4, 0], Card('3d'): [5, 0], Card('3h'): [6, 0], Card('3s'): [7, 0],
            Card('4c'): [8, 0], Card('4d'): [9, 0], Card('4h'): [10, 0], Card('4s'): [11, 0],
            Card('5c'): [12, 0], Card('5d'): [13, 0], Card('5h'): [14, 0], Card('5s'): [15, 0],
            Card('6c'): [16, 0], Card('6d'): [17, 0], Card('6h'): [18, 0], Card('6s'): [19, 0],
            Card('7c'): [20, 0], Card('7d'): [21, 0], Card('7h'): [22, 0], Card('7s'): [23, 0],
            Card('8c'): [24, 0], Card('8d'): [25, 0], Card('8h'): [26, 0], Card('8s'): [27, 0],
            Card('9c'): [28, 0], Card('9d'): [29, 0], Card('9h'): [30, 0], Card('9s'): [31, 0],
            Card("Tc"): [32, 0], Card("Td"): [33, 0], Card("Th"): [34, 0], Card("Ts"): [35, 0],
            Card("Jc"): [36, 0], Card("Jd"): [37, 0], Card("Jh"): [38, 0], Card("Js"): [39, 0],
            Card("Qc"): [40, 0], Card("Qd"): [41, 0], Card("Qh"): [42, 0], Card("Qs"): [43, 0],
            Card("Kc"): [44, 0], Card("Kd"): [45, 0], Card("Kh"): [46, 0], Card("Ks"): [47, 0],
            Card("Ac"): [48, 0], Card("Ad"): [49, 0], Card("Ah"): [50, 0], Card("As"): [51, 0]
        }

    def model_device(self):
        return next(self.model.parameters()).device

    def card_to_index(self, cards, max_cards_count) -> list:
        """
        Получаем уникальные индексы для карт
        """
        card_index_list = []
        for card in cards:
            card_index_list.append(self.card_dict[card][0])
        if len(cards) < max_cards_count:
            card_index_list.extend([52] * (max_cards_count - len(cards)))
        return tuple(card_index_list)

    def _norm_money(self, x):
        """
        Нормализация сумм (тождественна для bb=1)
        """
        def _one(v): return float(v)
        if isinstance(x, (list, tuple)):
            return [_one(v) for v in x]
        return _one(x)

    def get_action_from_index(self, action_index, game=None):
        game = game or self.game
        player = game.players[game.current_player]
        current_contrib = game.player_bet_amount(game.current_player)
        to_call = game.chips_to_call(game.current_player)
        if action_index == 0:
            return ActionType.FOLD, None
        if action_index == 1:
            return (ActionType.CHECK, None) if to_call == 0 else (ActionType.CALL, None)
        if 2 <= action_index <= 8:
            raise_index = action_index - 2
            try:
                min_raise_value = game.min_raise()
            except Exception:
                min_raise_value = self.big_blind
            if game.hand_phase == HandPhase.PREFLOP:
                fixed_option = [0, 0.5 * self.big_blind, 1.0 * self.big_blind,
                                1.5 * self.big_blind, 2.0 * self.big_blind,
                                2.5 * self.big_blind, 3.0 * self.big_blind]
                delta = fixed_option[raise_index]
            else:
                total_pot = sum(getattr(pot, 'amount', 0) for pot in game.pots)
                percentage_options = [0.25 * total_pot, 0.5 * total_pot, 0.75 * total_pot,
                                       1.0 * total_pot, 1.25 * total_pot, 1.5 * total_pot,
                                       1.75 * total_pot]
                delta = percentage_options[raise_index]
            raise_increment = max(min_raise_value, delta)
            target_total = current_contrib + to_call + raise_increment
            if target_total >= current_contrib + player.chips:
                return ActionType.ALL_IN, None
            return ActionType.RAISE, {'total': int(target_total)}
        if action_index == 9:
            return ActionType.ALL_IN, None
        return ActionType.CALL, None

    def build_action_mask(self, game, current_player):
        """
        Строим маску допустимых действий A=10 .
        Индексы действий:
        0: FOLD
        1: CHECK/CALL
        2 - 8: дискретные варианты RAISE
        9: ALL_IN
        """
        allowed = [0] * self.num_actions
        mi = game.get_available_moves()
        try:
            action_types = set(mi.action_types)
        except AttributeError:
            action_types = set(list(mi))
        if ActionType.FOLD in action_types:
            allowed[0] = 1
        if ActionType.CALL in action_types or ActionType.CHECK in action_types:
            allowed[1] = 1
        if ActionType.ALL_IN in action_types:
            allowed[9] = 1
        if ActionType.RAISE in action_types:
            for a_idx in range(2, 9):
                a_type, a_param = self.get_action_from_index(a_idx, game=game)
                if a_type != ActionType.RAISE:
                    continue
                total = None
                if isinstance(a_param, dict) and 'total' in a_param and a_param['total'] is not None:
                    total = int(a_param['total'])
                ok = False
                if total is not None:
                    try:
                        ok = bool(game.validate_move(action=ActionType.RAISE, total=total))
                    except Exception:
                        ok = False
                if ok:
                    allowed[a_idx] = 1
        return allowed

    def private_observation_space(self, current_player, game=None):
        """
        Возвращает всё пространство наблюдений для текущего игрока
        """
        game = game or self.game
        phase_list = [HandPhase.PREHAND, HandPhase.SETTLE, HandPhase.PREFLOP, HandPhase.FLOP, HandPhase.TURN, HandPhase.RIVER]
        phase = [i for i, ph in enumerate(phase_list) if game.hand_phase == ph][0]
        community_cards = self.card_to_index(game.board, 5)
        player_active = tuple(1 if p.state.name not in ("OUT", "SKIP") else 0 for p in game.players)
        player_bets_early = tuple(game.player_bet_amount(pid) for pid in range(self.num_players))
        pot_early = sum(getattr(pot, 'amount', 0) for pot in game.pots)
        player_chips_early = [p.chips for p in game.players]
        private_cards = self.card_to_index(game.get_hand(current_player), 2)
        to_call_early = game.chips_to_call(current_player)
        call_is_allin = int(to_call_early >= player_chips_early[current_player])
        allowed_actions = self.build_action_mask(game=game, current_player=current_player)
        pot = self._norm_money(pot_early)
        player_chips = self._norm_money(player_chips_early)
        player_bets = self._norm_money(player_bets_early)
        to_call = self._norm_money(to_call_early)
        seat_rel_actor = [((i - current_player) % self.num_players) for i in range(self.num_players)]
        if phase == 3:
            board_count = 3
        elif phase == 4:
            board_count = 4
        elif phase == 5:
            board_count = 5
        else:
            board_count = 0
        return {'phase': phase,
                'pot': pot,
                'community_cards': community_cards,
                'player_chips': player_chips,
                'player_bets': player_bets,
                'player_active': player_active,
                'current_player': current_player,
                'private_cards': private_cards,
                'allowed_actions': allowed_actions,
                'to_call': to_call,
                'call_is_allin': call_is_allin,
                'seat_rel_actor': seat_rel_actor,
                'board_count': board_count}

    def pack_to_tensor(self, x: dict):
        device = self.model_device()
        return {
            'phase': torch.tensor([x['phase']], dtype=torch.long, device=device),
            'pot': torch.tensor([float(x['pot'])], dtype=torch.float32, device=device),
            'community_cards': torch.tensor([list(x['community_cards'])], dtype=torch.long, device=device),
            'player_chips': torch.tensor([x['player_chips']], dtype=torch.float32, device=device),
            'player_bets': torch.tensor([x['player_bets']], dtype=torch.float32, device=device),
            'player_active': torch.tensor([x['player_active']], dtype=torch.float32, device=device),
            'current_player': torch.tensor([x['current_player']], dtype=torch.long, device=device),
            'private_cards': torch.tensor([list(x['private_cards'])], dtype=torch.long, device=device),
            'allowed_actions': torch.tensor([x['allowed_actions']], dtype=torch.float32, device=device),
            'to_call': torch.tensor([x['to_call']], dtype=torch.float32, device=device),
            'call_is_allin': torch.tensor([x['call_is_allin']], dtype=torch.float32, device=device),
            'seat_rel_actor': torch.tensor([x['seat_rel_actor']], dtype=torch.long, device=device),
            'board_count': torch.tensor([x['board_count']], dtype=torch.long, device=device)
        }

    def simulate_hand(self):
        self.game = TexasHoldEm(buyin=self.buyin, big_blind=self.big_blind, small_blind=self.small_blind, max_players=self.num_players)
        self.game.start_hand()
        episode_data = []
        while self.game.is_hand_running():
            current_player = self.game.current_player
            if current_player is None:
                break
            POS = self.private_observation_space(current_player, self.game)
            POS_t = self.pack_to_tensor(POS)
            with torch.no_grad():
                policy_probs, _ = self.model.forward(**POS_t)
            policy = policy_probs.cpu().numpy().flatten()
            valid = [i for i, f in enumerate(POS['allowed_actions']) if f]
            probs = [policy[i] if i in valid else 0.0 for i in range(self.num_actions)]
            s = sum(probs)
            probs = [p / s if s > 0 else (1/len(valid) if i in valid else 0) for i, p in enumerate(probs)]
            action_idx = random.choices(range(self.num_actions), weights=probs, k=1)[0]
            action_type, action_param = self.get_action_from_index(action_idx, self.game)
            try:
                if isinstance(action_param, dict):
                    self.game.take_action(action_type, **action_param)
                elif action_param is None:
                    self.game.take_action(action_type)
                else:
                    self.game.take_action(action_type, action_param)
            except Exception as e:
                print("Ошибка действия:", e, 'def simulate_hand')
                break
            state_features = POS
            action_mask = POS.pop('allowed_actions')
            episode_data.append({
                "state_features": state_features,
                "chosen_action": action_idx,
                "allowed_actions": action_mask
            })
        final_stacks = [p.chips for p in self.game.players]
        rewards = [final_stacks[i] - self.buyin for i in range(self.num_players)]
        return episode_data, rewards
