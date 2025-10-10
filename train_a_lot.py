#!/usr/bin/env python3
"""
    python train_a_lot.py --model chess_elo_model_V0 --dataset first_dataset_100k.npz --max_games 1048440 --game_split 1500 --max_file 100 --epochs 20 --start 1
"""
import argparse
import subprocess
import os


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True, help="model (es. chess_elo_model without .keras)")
    p.add_argument("--dataset", type=str, required=True, help="dataset .npz (es. dataset_from_fen.npz)")
    p.add_argument("--game_split", type=int, default=None, help="max number of games in ogni file")
    p.add_argument("--max_games", type=int, required=True, help="Numero massimo di partite da processare")
    p.add_argument("--max_file", type=int, default=1000, help="Numero massimo di file da processare")
    p.add_argument("--epochs", type=int, default=1, help="Max number of epochs to train for each file")
    p.add_argument("--start", type=int, default=1, help="File number to start")
    args = p.parse_args()

    base_dataset = os.path.splitext(args.dataset)[0]  # es: all_positions_jul2014_npz/positions_jul2014_npz

    start = 1 + (args.start - 1) * args.game_split
    batch = 1
    version = 1
    while batch <= args.max_file:
        end = min(start + args.game_split - 1, args.max_games)
        dataset_step = f"{base_dataset}_game{start}_game{end}.npz"
        model_save_to_step = f"model_versions/{args.model}_V{version}"

        cmd = [
            "python",
            "train_model.py",
            "--model", args.model,
            "--dataset", dataset_step,
            "--save_to", model_save_to_step,
            "--epochs", str(args.epochs),
        ]

        print(f"[Batch {batch}] Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

        start += args.game_split
        batch += 1
        version = batch / 10 + 1


if __name__ == "__main__":
    main()
