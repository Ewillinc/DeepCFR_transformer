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


from D_CFR_Transformer import DeepCFR
from Holdem_sim import HoldemSimulator

"""
-> random.seed(seed)
-> init TexasHoldem
-> start_hand

<------ ACTION_HISTORY_VECTOR

for action_index in action_history:
    action_type, action_param = get_action_from_index(action_index, game = game)

    if isinstance(action_param, dict):
        game.take_action(action_type, **action_param)
    else:
        game.take_action(action_type, action_param)
        
------->    return game


"""

import random
import logging
import numpy as np
import torch
from texasholdem.game.game import TexasHoldEm  


# Класс узла дерева игры, представляющий информационное состояние для traverser-игрока.
class GameTreeNode:
    def __init__(self, parent = None, action_from_parent = None):
        self.parent = parent                         # Ссылка на родителя (None = корень)
        self.action_from_parent = action_from_parent # Набор действий от родителя к этому узлу.

        self.children = {}                           # Каждый ребёнок == allowed_actions
        #        статы для sample
        self.state_features     = None              # Вектор наблюдения ( observation space )
        self.allowed_actions    = None              # Маска доступных действий в этом узле (allowed_actions)
        self.advantage_regrets  = None              # Целевые положительные регреты (r+) для всех действий
        self.target_policy      = None              # Целевое стратегическое распределение (pi по regret matching)
        self.utility            = None              # Полезность внутри узла U-mean ( из U - child)

# Класс для обхода дерева игры и сбора выборок для обучающих примеров.
class GameTreeSampler:
    def __init__(self, simulator):
        """
        Инициализация с симулятором игры (объект HoldemSimulator), который содержит:
        - Параметры игры (buyin, blinds, число игроков и т.д.)
        - Модель с методом model.forward для получения текущей стратегии (policy) и преимущества (advantage)
        - Метод get_action_from_index для перевода индекса действия (0-9) в реальное действие TexasHoldEm
        - Метод private_observation_space для получения признаков состояния и маски действий для игрока
        """
        self.sim = simulator
        self.root_node = None  # Корень построенного дерева для последней симуляции (узел traverser-игрока)

    def simulate_game_state(self, action_history, seed):
        """
        Воссоздаёт состояние игры, последовательно проигрывая все действия из `action_history`
        начиная с начала раздачи. Используется фиксированный `seed` для повторяемости расклада карт.
        Возвращает объект игры (TexasHoldEm) после применения всех действий из истории.
        """
        # Фиксируем генератор случайных чисел для детерминированного результата раздачи
        random.seed(seed)
        # Создаём новую игру TexasHoldEm с теми же параметрами, что и в исходном симуляторе
        game = TexasHoldEm(buyin=self.sim.buyin, big_blind=self.sim.big_blind,
                            small_blind=self.sim.small_blind, max_players=self.sim.num_players)
        game.start_hand()  # Начинаем новую раздачу (перетасовка колоды детерминирована фиксированным seed)
        # Последовательно выполняем каждое действие из истории на свежем экземпляре игры
        for a_idx in action_history:
            act_type, act_param = self.sim.get_action_from_index(a_idx, game=game)
            # Выполняем действие (если у действия есть параметр, передаём его правильно)
            if isinstance(act_param, dict):
                game.take_action(act_type, **act_param)
            else:
                game.take_action(act_type, act_param)
        return game

    def sample_policy_action(self, game, player_id):
        """
        Сэмплирует одно действие для игрока `player_id` (оппонента или chance) согласно текущей стратегии модели.
        Возвращает индекс выбранного действия `a_idx` на основе вероятностного распределения policy-head.
        """
        # Получаем признаки состояния и маску действий для данного игрока в текущем состоянии игры
        phase, pot, community_cards, player_chips, player_bets, player_active, cur_player, private_cards, allowed_actions = \
            self.sim.private_observation_space(player_id, game)
        state_features = (phase, pot, community_cards, player_chips, player_bets, player_active, cur_player, private_cards)
        mask = allowed_actions  # Маска допустимых действий (список из 0/1 длины self.sim.num_actions)
        # Пропускаем состояние через нейросеть, получая вероятности действий (policy_probs) для этого игрока.
        with torch.no_grad():
            policy_probs_tensor, _ = self.sim.model.forward(
                phase=torch.tensor([phase], dtype = torch.long),
                pot=torch.tensor([pot], dtype = torch.float32),
                community_cards=torch.tensor([list(community_cards)], dtype=torch.long),
                player_chips=torch.tensor([player_chips], dtype = torch.float32),
                player_bets=torch.tensor([player_bets], dtype = torch.float32),
                player_active=torch.tensor([player_active], dtype = torch.float32),
                current_player=torch.tensor([player_id], dtype = torch.long),
                private_cards=torch.tensor([list(private_cards)], dtype = torch.long),
                allowed_actions=torch.tensor([mask], dtype=torch.float32)
            )
        # Преобразуем тензор с вероятностями в numpy и применяем маску допустимых действий
        probs = policy_probs_tensor.cpu().numpy().flatten() * np.array(mask, dtype=float)
        # Если все вероятности оказались нулевыми (может случиться из-за числ. погрешностей), 
        # то распределяем равномерно по разрешённым действиям
        if probs.sum() == 0.0:
            probs = np.array(mask, dtype=float)
            if probs.sum() == 0.0:
                # Если вообще нет ни одного доступного действия (не должно происходить, хотя бы фолд должен быть),
                # возвращаем действие "фолд" (индекс 0) по умолчанию.
                return 0
            probs = probs / probs.sum()
        else:
            # Нормализуем распределение вероятностей на разрешённых действиях (сумма mask*probs = 1)
            probs = probs / probs.sum()
        # Сэмплируем индекс действия согласно полученному распределению вероятностей
        action_index = int(np.random.choice(len(probs), p=probs))
        logging.debug(f"Player {player_id} (opponent/chance) action distribution: {probs}, chosen action: {action_index}")
        return action_index

    def traverse_history(self, action_history, traverser_id, seed, parent_node=None, parent_action=None, samples_out=None):
        """
        Рекурсивно проходит по одному сценарию игры, ветвясь только в узлах текущего игрока `traverser_id`.
        Параметры:
          - action_history: список индексов действий от начала раздачи до текущего узла.
          - traverser_id: индекс игрока, для которого собираем опыт (текущий traverser).
          - seed: фиксированное случайное зерно, задающее расклад карт/борда для этой раздачи.
          - parent_node: родительский узел (GameTreeNode) в дереве игры для текущего состояния (None для корня).
          - parent_action: действие (индекс), приведшее к текущему узлу от родителя.
          - samples_out: список для накопления сгенерированных тренировочных примеров.
        Возвращает контрфактическую полезность (utility) для traverser из текущего узла.
        """
        # Воссоздаём состояние игры до текущего узла, проигрывая все действия из action_history на новой игре
        game = self.simulate_game_state(action_history, seed)
        # Базовый случай рекурсии: если игра (раздача) завершена, вычисляем и возвращаем итоговую выплату для traverser
        if not game.is_hand_running():
            final_stacks = [player.chips for player in game.players]
            traverser_utility = final_stacks[traverser_id] - self.sim.buyin  # выигрыш относительно стартового стека (бай-ина)
            logging.debug(f"Terminal node reached. Traverser {traverser_id} utility = {traverser_utility}")
            return traverser_utility

        # Узнаём, чей сейчас ход в восстановленном состоянии игры
        current_player = game.current_player

        # Если сейчас ход оппонента или случайного фактора (не traverser-игрока):
        if current_player != traverser_id:
            # Сэмплируем одно действие оппонента/шанса по текущей стратегии (external sampling)
            a_idx = self.sample_policy_action(game, current_player)
            logging.debug(f"Opponent/Chance node: player {current_player} took action {a_idx}")
            # Добавляем выбранное действие в историю и рекурсивно углубляемся дальше.
            # Заметьте: parent_node и parent_action не обновляются, потому что для traverser-узлов мы создаём новые узлы дерева,
            # а для оппонентских узлов мы просто продлеваем историю вглубь без создания нового узла (оппонентские узлы не записываются как обучающие образцы).
            return self.traverse_history(action_history + [a_idx], traverser_id, seed,
                                         parent_node=parent_node, parent_action=parent_action, samples_out=samples_out)

        # Если мы здесь, значит current_player == traverser_id — текущий узел является узлом решения traverser-игрока.
        # Создаём новый узел дерева для этого состояния и привязываем его к дереву (к родителю).
        print(f'parent_node {parent_node}')
        node = GameTreeNode(parent=parent_node, action_from_parent=parent_action)
        if parent_node is not None and parent_action is not None:
            # Привязываем новый узел как потомка родителя по действию parent_action
            parent_node.children[parent_action] = node
        else:
            # Если родителя нет (то есть это корень дерева), сохраняем этот узел как корневой
            self.root_node = node

        # Получаем текущее состояние (признаки + маску доступных действий) для traverser-игрока
        phase, pot, community_cards, player_chips, player_bets, player_active, cur_player, private_cards, allowed_actions = \
            self.sim.private_observation_space(current_player, game)
        state_features = (phase, pot, community_cards, player_chips, player_bets, player_active, cur_player, private_cards)
        mask = allowed_actions
        node.state_features = state_features
        node.allowed_actions = mask

        # Если по каким-то причинам нет доступных действий (маловероятно, обычно хотя бы фолд/чек доступны), возвращаем нулевую полезность
        if not any(mask):
            logging.debug(f"No allowed actions for traverser {traverser_id} at this state, returning 0 utility.")
            return 0.0




        #--------------------------------------------------------------------------------------------------------------

        #                       РЕКУРСИЯ ПО ВСЕМ ДЕЙСТВИЯМ В УЗЛЕ

        #--------------------------------------------------------------------------------------------------------------




        # Ветвление: перебираем все возможные действия traverser-игрока в этом узле.
        u_values = []  # список u(I→a) – контрфактическая полезность при выборе каждого действия a из этого состояния
        for action_index in range(self.sim.num_actions):
            if not mask[action_index]:
                # Если действие недоступно, его ценность не рассматривается (регрет будет 0)
                u_values.append(0.0)
                continue
            logging.debug(f"Traverser node: player {traverser_id} exploring action {action_index}")
            # Рекурсивно углубляемся по ветви с выбранным действием (traverser совершает action_index)
            u_val = self.traverse_history(action_history + [action_index], traverser_id, seed,
                                          parent_node=node, parent_action=action_index, samples_out=samples_out)
            u_values.append(u_val)
        u_values = np.array(u_values, dtype=np.float32)

        #--------------------------------------------------------------------------------------------------------------
        
        #--------------------------------------------------------------------------------------------------------------





        # Вычисляем среднюю полезность u_mean, если traverser играл бы согласно текущей стратегии (baseline value).
        # Здесь для простоты берём среднее по всем разрешённым действиям (это эквивалентно ожиданию при равномерной стратегии 
        # или при симметричной ситуации). В общем случае можно использовать вероятности текущей стратегии π_current.
        if any(mask):
            u_mean = u_values[np.array(mask, dtype=bool)].mean()  # среднее по ценностям доступных действий
        else:
            u_mean = 0.0

        # Вычисляем регреты: разница между ценностью каждого действия и базовой полезностью u_mean.
        regrets = u_values - u_mean
        # Оставляем только положительные регреты (отрицательные заменяем на 0, поскольку обучаем на положительных регретах).
        positive_regrets = np.clip(regrets, a_min=0.0, a_max=None)

        # Вычисляем целевое распределение стратегии (target_policy) через regret matching на положительных регретах.
        if positive_regrets.sum() > 1e-9:
            # Если есть положительные регреты, нормализуем их в вероятностное распределение.
            target_policy = (positive_regrets / positive_regrets.sum()).tolist()
        else:
            # Если все регреты неположительные (т.е. <= 0), устанавливаем равномерную стратегию на доступных действиях.
            count_allowed = sum(mask)
            target_policy = [1.0 / count_allowed if mask[i] else 0.0 for i in range(len(mask))]

        # Сохраняем вычисленные величины в текущем узле дерева (для возможного анализа или отладки)
        node.advantage_targets = positive_regrets.tolist()
        node.target_policy = target_policy
        node.utility = u_mean

        # Логируем рассчитанные параметры узла (для отладки)
        logging.info(f"Traverser node: allowed_actions={mask}, u_mean={u_mean:.3f}")
        logging.info(f"Regrets: {regrets.tolist()} | Positive regrets: {node.advantage_targets}")
        logging.info(f"Target policy (normalized positive regrets): {node.target_policy}")

        # Формируем обучающий пример из этого узла и добавляем в выходной список samples_out.
        sample = {
            "state_features": state_features,
            "advantage_targets": node.advantage_targets,
            "target_policy": node.target_policy,
            "allowed_actions": mask,
        }
        if samples_out is not None:
            samples_out.append(sample)

        # Возвращаем ожидаемую полезность u_mean этого узла – она используется для расчёта регретов на уровне выше.
        return u_mean

    def generate_samples(self, traverser_id, max_terminals=1):
        """
        Генерирует обучающие примеры (samples) через один обход игры за указанного traverser-игрока.
        Параметры:
          - traverser_id: индекс игрока, для которого собираем примеры (т.е. обучаем его стратегию).
          - max_terminals: максимальное число терминальных узлов для остановки (гиперпараметр, по умолчанию 1 полная раздача).
        Возвращает список словарей-примеров, собранных за прохождение (каждый содержит признаки состояния и целевые метрики).
        """
        samples_out = []
        # Фиксируем случайное зерно для расклада карт в этой раздаче (делаем раздачу детерминированной)
        seed = random.randrange(1, 2**32)
        logging.info(f"Starting traversal for traverser player {traverser_id} with seed={seed}")
        # Запускаем рекурсивный обход игры с пустой историей действий
        self.root_node = None
        self.traverse_history([], traverser_id, seed, parent_node=None, parent_action=None, samples_out=samples_out)
        # (Опционально) Можно прерывать дальнейший обход, если достигнуто max_terminals терминальных узлов.
        # Здесь max_terminals=1 означает, что мы проходим одну полную раздачу (до терминального узла).
        return samples_out


# %%
# %% [markdown]
# # 1. Подключаем библиотеки и модуль с обходом DeepCFR

import logging
"""from texasholdem.game.game import TexasHoldEm
from for_gpt_traverse_1 import HoldemSimulator, GameTreeSampler"""

# %% [markdown]
# # 2. Настраиваем логирование (опционально)

logging.basicConfig(level=logging.INFO, format="%(message)s")


# %%
# %% [markdown]
# # 3. Создаём симулятор и sampler

# 2 игрока для простоты, бай-ин 20000, блайнды 100/50
sim = HoldemSimulator(num_players=2, buyin=20000, big_blind=100, small_blind=50)
sampler = GameTreeSampler(sim)


# %%
# %% [markdown]
# # 4. Запускаем traversal для одной раздачи

traverser_id = 0         # обучаем стратегию первого игрока
samples = sampler.generate_samples(traverser_id)


# %%
# %% [markdown]
# # 5. Смотрим результаты

print(f"Всего сгенерировано узлов: {len(samples)}\n")

# Показ первых 5 примеров
for idx, sample in enumerate(samples[:5], start=1):
    print(f"--- Sample #{idx} ---")
    print("Features:", sample["state_features"])
    print("Allowed:", sample["allowed_actions"])
    print("r+     :", sample["advantage_targets"])
    print("π*     :", sample["target_policy"])
    print()



