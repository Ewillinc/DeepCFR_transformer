# %%
import torch
import torch.nn as nn
import torch.nn.functional as F


import numpy as np
import math, random, os

from texasholdem.game.game import TexasHoldEm
from texasholdem.game.action_type import ActionType
from texasholdem.game.hand_phase import HandPhase
from texasholdem.card.card import Card
from texasholdem.game.player_state import PlayerState

from D_CFR_Train import DeepCFR


class HoldemSimulator:
    def __init__(self, num_players=8, buyin=20000, big_blind=100,
                 small_blind=50, **model_params):
        """
        Симулятор для TexasHoldem, взаимодействующий с DeepCFR_transformer
        """

        self.num_players = num_players
        self.buyin = buyin
        self.big_blind = big_blind
        self.small_blind = small_blind

        self.num_actions = 10


        # Инициализируем игровой стол и модель
        self.game = TexasHoldEm(buyin=buyin, big_blind=big_blind, small_blind=small_blind, max_players=num_players)
        self.model = DeepCFR(num_players=num_players, **model_params).to('cpu')
        self.card_dict = {  Card('2c'):[0,0],Card('2d'):[1,0],Card('2h'):[2,0],Card('2s'):[3,0],
                            Card('3c'):[4,0],Card('3d'):[5,0],Card('3h'):[6,0],Card('3s'):[7,0],
                            Card('4c'):[8,0],Card('4d'):[9,0],Card('4h'): [10,0],Card('4s'):[11,0],
                            Card('5c'):[12,0],Card('5d'):[13,0],Card('5h'):[14,0],Card('5s'):[15,0],
                            Card('6c'):[16,0],Card('6d'):[17,0],Card('6h'):[18,0],Card('6s'):[19,0],
                            Card('7c'):[20,0],Card('7d'):[21,0],Card('7h'):[22,0],Card('7s'):[23,0],
                            Card('8c'):[24,0],Card('8d'):[25,0],Card('8h'):[26,0],Card('8s'):[27,0],
                            Card('9c'):[28,0],Card('9d'):[29,0],Card('9h'):[30,0],Card('9s'):[31,0],
                            Card("Tc"):[32,0],Card("Td"):[33,0],Card("Th"):[34,0],Card("Ts"):[35,0],
                            Card("Jc"):[36,0],Card("Jd"):[37,0],Card("Jh"):[38,0],Card("Js"):[39,0],
                            Card("Qc"):[40,0],Card("Qd"):[41,0],Card("Qh"):[42,0],Card("Qs"):[43,0],
                            Card("Kc"):[44,0],Card("Kd"):[45,0],Card("Kh"):[46,0],Card("Ks"):[47,0],
                            Card("Ac"):[48,0],Card("Ad"):[49,0],Card("Ah"):[50,0],Card("As"):[51,0]}

    # ----------------------------------------------------------------
    # 0. Преобразует карты в уникальный id,
    # ----------------------------------------------------------------

    def card_to_index(self,cards, max_cards_count) -> list:
        """
        получаем уникальные индексы для карт. Ввиду внутренних особеннойстей 
        texasholdem 0.11. 
        У игры нету точно сопоставляемых индексов для каждой карты, масти и 
        значения отдельно. Было проще сделать этот словарь и обращаться всегда
        """
        card_index_list = []
        for card in cards:
            card_index_list.append(self.card_dict[card][0])
        if len(cards) < max_cards_count:
            card_index_list.extend([52] * (max_cards_count - len(cards)))
        return tuple(card_index_list)


    # ----------------------------------------------------------------
    # 1. индекс ->  действие  
    # ----------------------------------------------------------------
    def get_action_from_index(self, action_index, game=None):

        """
        Возвращает действие которое должен совершить агент,
        соответствующая индексу этого действия и политики принятия
        решения.
    
        Для PREFLOP      - ставки фиксированы объемом от BB
        Для FLOP и далее - процент от POT
        
        """



        game = game or self.game                     # ← ключевая строка
        player = game.players[game.current_player]
        current_contrib = game.player_bet_amount(game.current_player)
        to_call         = game.chips_to_call(game.current_player)

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
                fixed_option = [0,
                                0.5 * self.big_blind,
                                1.0 * self.big_blind,
                                1.5 * self.big_blind,
                                2.0 * self.big_blind,
                                2.5 * self.big_blind,
                                3.0 * self.big_blind]
                delta = fixed_option[raise_index]
            else:
                total_pot = sum(getattr(pot, 'amount', 0) for pot in game.pots)
                percentage_options = [0.25 * total_pot, 0.5 * total_pot, 0.75 * total_pot,
                                      1.0  * total_pot, 1.25 * total_pot, 1.5 * total_pot,
                                      1.75 * total_pot]
                delta = percentage_options[raise_index]

            raise_increment = max(min_raise_value, delta)
            target_total    = current_contrib + to_call + raise_increment

            if target_total >= current_contrib + player.chips:
                return ActionType.ALL_IN, None
            return ActionType.RAISE, {'total': int(target_total)}

        if action_index == 9:
            return ActionType.ALL_IN, None
        return ActionType.CALL, None

    #-----------------------------------------------------------------
    # 2. Создает вектор Observation_Space с публичной и частной информацией
    # ----------------------------------------------------------------


    def private_observation_space(self, current_player, game=None):

        """
        Возвращает вектор Observation_space, представляющий из себя список 
        уникальных наблюдений для каждого игрока частной и общей информации
        во время раздачи.

        Модель делает свои предсказания опираясь на это вектор
        """


        game = game or self.game

        phase_list = [HandPhase.PREHAND, HandPhase.SETTLE,
                      HandPhase.PREFLOP, HandPhase.FLOP,
                      HandPhase.TURN, HandPhase.RIVER]
        phase = [i for i, ph in enumerate(phase_list) if game.hand_phase == ph][0]

        community_cards = self.card_to_index(game.board, 5)
        player_active   = tuple(1 if p.state.name not in ("OUT", "SKIP") else 0
                                for p in game.players)
        player_bets     = tuple(game.player_bet_amount(pid) for pid in range(self.num_players))
        pot             = sum(getattr(pot, 'amount', 0) for pot in game.pots)
        player_chips    = [p.chips for p in game.players]
        private_cards   = self.card_to_index(game.get_hand(current_player), 2)

        # ――― маска действий ―――
        legal_actions   = list(game.get_available_moves())
        if len(legal_actions) == 1:
            allowed_actions = [0]*self.num_actions
            allowed_actions[0] = allowed_actions[1] = 1
        else:
            allowed_actions = [0]*self.num_actions
            allowed_actions[0] = allowed_actions[1] = 1
            to_call       = game.chips_to_call(current_player)
            player_obj    = game.players[current_player]
            if player_obj.chips > to_call:
                try:
                    min_raise_amt = game.min_raise()
                except Exception:
                    min_raise_amt = self.big_blind
                current_contrib = game.player_bet_amount(current_player)
                for act_idx in range(2, self.num_actions):
                    act_type, param = self.get_action_from_index(act_idx, game)
                    if act_type == ActionType.RAISE:
                        target_total = param.get('total') if isinstance(param, dict) else None
                        if target_total - current_contrib >= min_raise_amt + to_call and \
                           target_total <= current_contrib + player_obj.chips:
                            allowed_actions[act_idx] = 1
                    elif act_type == ActionType.ALL_IN:
                        allowed_actions[act_idx] = 1

        return (phase, pot, community_cards, player_chips,
                player_bets, player_active, current_player,
                private_cards, allowed_actions)

    # ----------------------------------------------------------------
    #  Симулирует одну раздачу
    # ----------------------------------------------------------------

    def simulate_hand(self):
        """
        Симулируем одну раздачу с политикой принятия действий нейронной сетью.
        """
        self.game = TexasHoldEm(buyin=self.buyin, big_blind=self.big_blind,
                                small_blind=self.small_blind, max_players=self.num_players)
        self.game.start_hand()
        episode_data = []

        while self.game.is_hand_running():
            current_player = self.game.current_player
            if current_player is None:
                break

            POS = self.private_observation_space(current_player, self.game)

            with torch.no_grad():
                policy_probs, _ = self.model.forward(
                    phase=POS[0], pot=POS[1],
                    community_cards=list(POS[2]),
                    player_chips=POS[3], player_bets=POS[4],
                    player_active=POS[5], current_player=POS[6],
                    private_cards=list(POS[7]),
                    allowed_actions=POS[8]
                )

            policy = policy_probs.cpu().numpy().flatten()
            valid  = [i for i, f in enumerate(POS[8]) if f]
            probs  = [policy[i] if i in valid else 0.0 for i in range(self.num_actions)]
            s      = sum(probs)
            probs  = [p / s for p in probs] if s > 0 else [1/len(valid) if i in valid else 0 for i in range(self.num_actions)]
            action_idx = random.choices(range(self.num_actions), weights=probs, k=1)[0]

            action_type, action_param = self.get_action_from_index(action_idx, self.game)
            try:
                self.game.take_action(action_type, **action_param) \
                    if isinstance(action_param, dict) else self.game.take_action(action_type, action_param)
            except Exception as e:
                print("Ошибка действия:", e)
                break

            episode_data.append({
                "state_features": (POS[0], POS[1], list(POS[2]),
                                   POS[3], POS[4], POS[5],
                                   POS[6], tuple(POS[7])),
                "chosen_action": action_idx,
                "allowed_actions": POS[8]
            })

        final_stacks = [p.chips for p in self.game.players]
        rewards      = [final_stacks[i] - self.buyin for i in range(self.num_players)]
        return episode_data, rewards
