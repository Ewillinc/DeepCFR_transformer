from __future__ import annotations
import os
import sys
import argparse
import random
from typing import List
import numpy as np
import torch
from holdem_simulator import HoldemSimulator
from traverse_buffers import GameTreeSampler, StageBuffers
from deepcfr_trainer import DeepCFRTrainer

def parse_args(argv: List[str]):
    p = argparse.ArgumentParser(description="DeepCFR training loop")

    p.add_argument("--device", default="cuda", type=str)
    p.add_argument("--logdir", default="./runs/dcfr", type=str)
    p.add_argument("--ckpt_dir", default="./checkpoints", type=str)
    p.add_argument("--resume", default=None, type=str)

    p.add_argument("--vector_players", default=8, type=int)
    p.add_argument("--active_players", default=2, type=int)
    p.add_argument("--buyin", default=200, type=int)
    p.add_argument("--big_blind", default=1, type=int)
    p.add_argument("--small_blind", default=0.5, type=int)
    p.add_argument("--card_emb_dim", default=32, type=int)
    p.add_argument("--player_emb_dim", default=16, type=int)
    p.add_argument("--hidden_dim", default=128, type=int)
    p.add_argument("--n_attention_heads", default=4, type=int)
    p.add_argument("--n_attention_layers", default=2, type=int)
    p.add_argument("--num_actions", default=10, type=int)

    p.add_argument("--lr", default=3e-4, type=float)
    p.add_argument("--N_nodes", default=5000, type=int)
    p.add_argument("--max_hands", default=200, type=int)
    p.add_argument("--M_updates", default=2000, type=int)
    p.add_argument("--batch_size", default=512, type=int)
    p.add_argument("--cycles", default=20, type=int)
    p.add_argument("--seed", default=42, type=int)
    return p.parse_args(argv)

def _load_latest_checkpoint(ckpt_dir: str):
    if not os.path.isdir(ckpt_dir):
        return None
    files = [f for f in os.listdir(ckpt_dir) if f.endswith((".pt", ".pth"))]
    if not files:
        return None
    files.sort()
    return os.path.join(ckpt_dir, files[-1])

def _total_in_buffers(stage_buffers: StageBuffers):
    return sum(len(lst) for lst in stage_buffers.by_stage.values())

def main(argv: List[str]):
    args = parse_args(argv)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.logdir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    device = torch.device(args.device)

    sim = HoldemSimulator(
        num_players=args.vector_players,
        buyin=args.buyin,
        big_blind=args.big_blind,
        small_blind=max(1, args.big_blind // 2),
        normalize_money=True,
        card_emb_dim=args.card_emb_dim,
        player_emb_dim=args.player_emb_dim,
        hidden_dim=args.hidden_dim,
        n_attention_heads=args.n_attention_heads,
        n_attention_layers=args.n_attention_layers,
        num_actions=args.num_actions,
    )
    sim.model.to(device)
    sampler = GameTreeSampler(simulator=sim, global_sample_limit=int(args.N_nodes))
    trainer = DeepCFRTrainer(
        simulator=sim,
        sampler=sampler,
        batch_size=int(args.batch_size),
        adv_steps=int(args.M_updates),
        pol_steps=int(args.M_updates),
        lr=float(args.lr),
        tb_logdir=args.logdir,
    )

    if args.resume:
        ckpt_path = args.resume
        if args.resume == "latest":
            maybe = _load_latest_checkpoint(args.ckpt_dir)
            if maybe is not None:
                ckpt_path = maybe
        if ckpt_path and os.path.isfile(ckpt_path):
            print(f" loading weights from {ckpt_path}")
            state = torch.load(ckpt_path, map_location=device)
            sim.model.load_state_dict(state)
        else:
            print(f"checkpoint not found: {ckpt_path}")
    print(f" device={device}, players={args.vector_players}, active_players(rotating traverser)={args.active_players}")

    for cycle in range(1, args.cycles + 1):
        
        sampler.stage_buffers = StageBuffers(per_stage_limit=None)
        sampler.global_sample_limit = int(args.N_nodes)
        
        hands = 0
        while True:
            traverser_id = hands % max(1, int(args.active_players))
            _samples, _by_stage = sampler.generate_samples(traverser_id=traverser_id, fixed_seed=None)
            hands += 1
            total_now = _total_in_buffers(sampler.stage_buffers)
            if total_now >= int(args.N_nodes) or hands >= int(args.max_hands):
                break
        total_collected = _total_in_buffers(sampler.stage_buffers)
        print(f"цикл: {cycle} collected {total_collected} samples in {hands} hands")
        adv_loss, pol_loss = trainer.train_on_buffer()
        print(f"цикл: {cycle} losses: adv_loss={adv_loss:.6f}  policy_loss={pol_loss:.6f}")
        ckpt_path = os.path.join(args.ckpt_dir, f"dcfr_cycle_{cycle:04d}.pt")
        torch.save(sim.model.state_dict(), ckpt_path)
        print(f"цикл: {cycle} checkpoint saved to: {ckpt_path}")
    print("Обучение end")

if __name__ == "__main__":
    main(sys.argv[1:])
