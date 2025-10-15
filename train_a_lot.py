#!/usr/bin/env python3
"""
    python train_a_lot.py --model model_versions/chess_elo_model --dataset all_positions_jul2014_npz/positions_jul2014.npz --max_games 1048440 --game_split 1500 --max_file 100 --epochs 10 --validation validation_100k_positions_from_130_files.npz

"""
import argparse
import subprocess
import os
from datetime import datetime

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True, help="model (es. chess_elo_model without .keras)")
    p.add_argument("--dataset", type=str, required=True, help="dataset .npz (es. dataset_from_fen.npz)")
    p.add_argument("--game_split", type=int, default=None, help="max number of games in ogni file")
    p.add_argument("--max_games", type=int, required=True, help="Numero massimo di partite da processare")
    p.add_argument("--max_file", type=int, default=1000, help="Numero massimo di file da processare")
    p.add_argument("--epochs", type=int, default=1, help="Max number of epochs to train for each file")
    p.add_argument("--start", type=int, default=1, help="File number to start")
    p.add_argument("--starting_version", type=int, default=0)
    p.add_argument("--validation", default=None, help="validation dataset")
    args = p.parse_args()

    base_dataset = os.path.splitext(args.dataset)[0]  # es: all_positions_jul2014_npz/positions_jul2014

    start = 1 + (args.start - 1) * args.game_split
    batch = 1
    version = args.starting_version
    model_old = f"{args.model}_V{version}"
    while batch <= args.max_file:
        end = min(start + args.game_split - 1, args.max_games)
        dataset_step = f"{base_dataset}_game{start}_game{end}.npz"
        model_save_to_step = f"{args.model}_V{int(version + 1)}"

        cmd = [
            "python",
            "train_model.py",
            "--model", model_old,
            "--dataset", dataset_step,
            "--save_to", model_save_to_step,
            "--epochs", str(args.epochs),
            "--validation", args.validation
        ]

        time = datetime.now().isoformat()
        tot_com = f"Time: {time}, Start:{start}, End:{end}, Version:{version}\n[Batch {batch}] Running: {' '.join(cmd)}"
        print(tot_com)
        with open("log_train.txt", "a", encoding="utf-8") as f:
            f.write(f"{tot_com}\n\n")
        subprocess.run(cmd, check=True)

        model_old = model_save_to_step
        version = batch / 10 + args.starting_version
        start += args.game_split
        batch += 1


if __name__ == "__main__":
    main()
