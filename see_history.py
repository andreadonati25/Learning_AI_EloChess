#!/usr/bin/env python3
"""
see_history.py

Legge un file .npy creato da train_model.py (lista di entry) e produce un file .txt leggibile.

Usage:
  python see_history.py --history model_versions/chess_elo_model_V1_history.npy
"""
import argparse
import numpy as np
from pathlib import Path
from pprint import pformat

def pretty_list(vals, max_items=200):
    # converte una lista numerica in stringa leggibile, tronca se troppo lunga
    s = ", ".join(f"{float(v):.4f}" for v in vals)
    if len(vals) > max_items:
        # tronca con indicazione
        s = ", ".join(f"{float(v):.4f}" for v in vals[:max_items]) + ", ... (troncato)"
    return s

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--history", required=True, help="path al file .npy con la history (es. chess_elo_model_V1_history.npy)")
    p.add_argument("--out", help="path output .txt (default sameprefix_history.txt)")
    args = p.parse_args()

    hist_path = Path(args.history)
    if not hist_path.exists():
        print("File non trovato:", hist_path)
        return

    # output path default
    if args.out:
        out_path = Path(args.out)
    else:
        out_path = hist_path.with_suffix("")  # rimuove .npy
        out_path = Path(str(out_path) + ".txt")

    data = np.load(str(hist_path), allow_pickle=True)
    # converti in lista python
    if isinstance(data, np.ndarray):
        entries = data.tolist()
    else:
        entries = data

    if isinstance(entries, dict):
        entries = [entries]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"History file: {hist_path}\n")
        f.write(f"Total runs: {len(entries)}\n")
        f.write("="*80 + "\n\n")
        for i, entry in enumerate(entries):
            f.write(f"RUN #{i+1}\n")
            f.write("-"*60 + "\n")
            ts = entry.get("timestamp_utc", "N/A")
            f.write(f"Timestamp (UTC): {ts}\n")

            # args
            args_map = entry.get("args", {})
            f.write("Args (salvati):\n")
            if isinstance(args_map, dict):
                for k, v in sorted(args_map.items()):
                    f.write(f"  {k}: {v}\n")
            else:
                f.write(f"  {pformat(args_map)}\n")
            f.write("\n")

            # best
            best = entry.get("best", {})
            if best:
                f.write("Best per metrica (valore, epoca, mode):\n")
                for k, info in sorted(best.items()):
                    bv = info.get("best_value")
                    ep = info.get("epoch")
                    mode = info.get("mode", "")
                    f.write(f"  {k}: {bv:.6f}   (epoch {ep})   mode={mode}\n")
            else:
                f.write("Best: (non disponibile)\n")
            f.write("\n")

            # full history
            hist = entry.get("history", {})
            if hist:
                f.write("Full history (per metrica -> lista di valori per epoca):\n")
                for k, vals in sorted(hist.items()):
                    try:
                        svals = pretty_list(vals)
                    except Exception:
                        svals = pformat(vals)
                    f.write(f"  {k} : [{svals}]\n")
            else:
                f.write("History: (non disponibile)\n")

            f.write("\n" + "="*80 + "\n\n")

    print(f"Esportazione completata: {out_path}")

if __name__ == "__main__":
    main()
