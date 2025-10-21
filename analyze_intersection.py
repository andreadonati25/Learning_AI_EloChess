#!/usr/bin/env python3
"""
analyze_intersection.py

Analisi avanzata: intersezione top-K (Model vs Stockfish) + posizione della
migliore mossa di Stockfish (top-1 SF) nella classifica del modello.

Modifiche:
 - salva tutti i file di output in una cartella chiamata come --out_prefix
 - stampa riepilogo con indicazione della cartella di output

Example:
    python analyze_intersection.py --csv Validation_Examples_V1.csv --out_prefix results_V1 --save_zero_examples
"""
import argparse
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import os
import json

def parse_list_cell(cell):
    if pd.isna(cell):
        return []
    if isinstance(cell, (list, tuple)):
        return [str(x).strip() for x in cell if str(x).strip() != ""]
    s = str(cell).strip()
    if s == "":
        return []
    if ";" in s and "," in s:
        parts = [p.strip() for p in s.split(";") if p.strip() != ""]
    elif ";" in s:
        parts = [p.strip() for p in s.split(";") if p.strip() != ""]
    else:
        parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    clean = []
    for p in parts:
        if ":" in p:
            move = p.split(":", 1)[0].strip()
        else:
            move = p.strip()
        move = move.strip("'\" ")
        if move != "":
            clean.append(move)
    return clean

def compute_intersections(df, col_model="topk", col_sf="topk_stockfish", ks=(5,)):
    model_lists = df[col_model].apply(parse_list_cell).tolist()
    sf_lists = df[col_sf].apply(parse_list_cell).tolist()
    N = len(model_lists)
    results = {}
    for K in ks:
        counts = Counter()
        per_sample = []
        for mlist, slist in zip(model_lists, sf_lists):
            m_topk = mlist[:K]
            s_topk = slist[:K]
            inter = set(m_topk).intersection(s_topk)
            c = len(inter)
            counts[c] += 1
            per_sample.append(c)
        freq = [counts.get(i, 0) for i in range(0, K+1)]
        results[K] = {"freq": freq, "counts": counts, "N": N, "per_sample": per_sample}
    return results

def compute_sf_best_rank(df, col_model="topk", col_sf="topk_stockfish", max_rank=10):
    model_lists = df[col_model].apply(parse_list_cell).tolist()
    sf_lists = df[col_sf].apply(parse_list_cell).tolist()
    N = len(model_lists)

    rank_counter = Counter()
    ranks_list = []
    for mlist, slist in zip(model_lists, sf_lists):
        if len(slist) == 0:
            rank_counter["no_sf"] += 1
            ranks_list.append(None)
            continue
        sf_best = slist[0]
        found_rank = None
        try:
            found_rank = mlist.index(sf_best)
        except Exception:
            found_rank = None
        if found_rank is None:
            rank_counter["not_in_model"] += 1
        else:
            if found_rank < max_rank:
                rank_counter[int(found_rank)] += 1
            else:
                rank_counter[f"rank>{max_rank-1}"] += 1
        ranks_list.append(found_rank)
    return {"counter": rank_counter, "ranks_list": ranks_list, "N": N}

def plot_intersection_bar(freq, K, N, outpath=None, title=None):
    xs = list(range(0, K+1))
    ys = freq
    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(xs, ys)
    ax.set_xticks(xs)
    ax.set_xlabel("Dimensione intersezione (0..K)")
    ax.set_ylabel("Numero esempi")
    if title:
        ax.set_title(title)
    for x, y in zip(xs, ys):
        pct = 100.0 * y / N if N>0 else 0.0
        ax.text(x, y + max(1, N*0.005), f"{y}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    if outpath:
        fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)

def plot_intersection_cumulative(freq, K, N, outpath=None, title=None):
    xs = list(range(1, K+1))
    cum = []
    total = sum(freq)
    for t in xs:
        ge_t = sum(v for i, v in enumerate(freq) if i >= t)
        cum.append(100.0 * ge_t / total if total>0 else 0.0)
    fig, ax = plt.subplots(figsize=(7,4))
    ax.plot(xs, cum, marker='o')
    ax.set_xticks(xs)
    ax.set_xlabel("Soglia t (intersezione >= t)")
    ax.set_ylabel("Percentuale esempi (%)")
    if title:
        ax.set_title(title)
    for x, y in zip(xs, cum):
        ax.text(x, y + max(0.5, N*0.001), f"{y:.1f}%", ha="center")
    ax.grid(linestyle="--", alpha=0.3)
    if outpath:
        fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)

def plot_sf_rank_hist(counter, max_rank, outpath=None, title=None):
    keys = [i for i in range(max_rank)]
    labels = []
    values = []
    for k in keys:
        labels.append(str(k))
        values.append(counter.get(k, 0))
    if counter.get(f"rank>{max_rank-1}", 0) > 0:
        labels.append(f">{max_rank-1}")
        values.append(counter.get(f"rank>{max_rank-1}", 0))
    for name in ("not_in_model", "no_sf"):
        if counter.get(name, 0) > 0:
            labels.append(name)
            values.append(counter.get(name, 0))
    N = sum(values)
    fig, ax = plt.subplots(figsize=(9,4))
    xs = list(range(len(values)))
    ax.bar(xs, values)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlabel("Rank della best-move SF nella ranking del modello (0 = top1 model)")
    ax.set_ylabel("Numero esempi")
    if title:
        ax.set_title(title)
    for x, y in zip(xs, values):
        pct = 100.0 * y / N if N>0 else 0.0
        ax.text(x, y + max(1, N*0.005), f"{y}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    if outpath:
        fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="CSV prodotto da lets_try.py (delim ';' o ',')")
    p.add_argument("--col_model", default="topk", help="nome colonna con top-K del modello")
    p.add_argument("--col_sf", default="topk_stockfish", help="nome colonna con top-K di Stockfish")
    p.add_argument("--max_rank", type=int, default=5, help="massimo rank da considerare per l'analisi SF-best")
    p.add_argument("--out_prefix", default="analysis", help="nome della cartella di output dove salvare grafici e summary")
    p.add_argument("--elo_col", default="elo_side", help="nome colonna che contiene l'elo del side (usata per bucket)")
    p.add_argument("--save_zero_examples", action="store_true", help="salva gli esempi con intersezione==0 in CSV")
    args = p.parse_args()

    # crea cartella di output
    out_dir = args.out_prefix
    os.makedirs(out_dir, exist_ok=True)

    # lettura CSV (prova ; poi ,)
    try:
        df = pd.read_csv(args.csv, delimiter=";", encoding="utf-8")
    except Exception:
        df = pd.read_csv(args.csv, delimiter=",", encoding="utf-8")

    if args.col_model not in df.columns or args.col_sf not in df.columns:
        print("Colonne richieste non trovate nel CSV. Colonne disponibili:", df.columns.tolist())
        return

    # SF-best analysis (global)
    sf_rank = compute_sf_best_rank(df, col_model=args.col_model, col_sf=args.col_sf, max_rank=args.max_rank)
    counter = sf_rank["counter"]
    out_rank = os.path.join(out_dir, f"sf_best_rank.png")
    plot_sf_rank_hist(counter, args.max_rank, outpath=out_rank, title=f"Rank di stockfish-top1 nella top-{args.max_rank} del modello")

    ks = [3,5]
    inter_results = compute_intersections(df, col_model=args.col_model, col_sf=args.col_sf, ks=ks)

    summary = {}
    for K in ks :
        freq = inter_results[K]["freq"]
        N = inter_results[K]["N"]
        out_bar = os.path.join(out_dir, f"intersection_K{K}.png")
        out_cum = os.path.join(out_dir, f"intersection_cumulative_K{K}.png")
        title = f"Intersection top-{K} Model vs Stockfish (N={N})"
        plot_intersection_bar(freq, K, N, outpath=out_bar, title=title)
        plot_intersection_cumulative(freq, K, N, outpath=out_cum, title=f"{title} - cumulativa")

        # stampa riepilogo sintetico su stdout (con la formulazione richiesta)
        print(f"\n=== Riepilogo generale K = {K} ===")
        print(f"Numero esempi analizzati: {N}")
        freq = inter_results[K]["freq"]
        print(f"\nTop-{K} intersection distribution (counts, perc):")
        for i, cnt in enumerate(freq):
            print(f"  inter={i:2d} : {cnt:6d} ({100.0*cnt/N:.2f}%)")
        at_least_one = sum(freq[1:]) if len(freq)>1 else 0
        print(f"  esempi con intersezione >=1: {at_least_one} ({100.0*at_least_one/N:.2f}%)")

        with open(os.path.join(out_dir, f"riepilogo_generale_{K}.txt"), "w", encoding="utf-8") as f:
            f.write(f"\n=== Riepilogo generale K = {K} ===\n")
            f.write(f"Numero esempi analizzati: {N}\n")
            f.write(f"\nTop-{K} intersection distribution (counts, perc):\n")
            for i, cnt in enumerate(freq):
                f.write(f"  inter={i:2d} : {cnt:6d} ({100.0*cnt/N:.2f}%)\n")
            f.write(f"  esempi con intersezione >=1: {at_least_one} ({100.0*at_least_one/N:.2f}%)\n")

        # Summary JSON (global)
        summary[f"intersection_{K}"] = {
                str(K): {
                    "counts": {str(i): int(v) for i, v in enumerate(freq)},
                    "percentages": {str(i): 100.0 * int(v) / N if N>0 else 0.0 for i, v in enumerate(freq)}
                }
            }

        # save zero-intersection examples
        per_sample_counts = np.array(inter_results[K]["per_sample"], dtype=int)
        zero_idxs = np.where(per_sample_counts == 0)[0]
        if args.save_zero_examples and len(zero_idxs) > 0:
            out_zero = os.path.join(out_dir, f"zero_intersection_top{K}_examples.csv")
            df_zero = df.iloc[zero_idxs]
            df_zero.to_csv(out_zero, index=False)
            print(f"Saved {len(zero_idxs)} zero-intersection in top-{K} examples to {out_zero}")

    # percentuali top1/top3/top5 per SF-best presence in model
    top1 = counter.get(0, 0)
    top3 = sum(counter.get(i,0) for i in range(0, min(3, args.max_rank)))
    top5 = sum(counter.get(i,0) for i in range(0, min(5, args.max_rank)))
    not_in = counter.get("not_in_model", 0)
    no_sf = counter.get("no_sf", 0)
    summary[f"sf_best"] = {"counter": dict(counter)}
    summary["sf_best"]["top1"] = {"count": int(top1), "pct": 100.0*top1/N if N>0 else 0.0}
    summary["sf_best"]["top3"] = {"count": int(top3), "pct": 100.0*top3/N if N>0 else 0.0}
    summary["sf_best"]["top5"] = {"count": int(top5), "pct": 100.0*top5/N if N>0 else 0.0}
    summary["sf_best"]["not_in_model"] = int(not_in)
    summary["sf_best"]["no_sf"] = int(no_sf)

    ranks = [r for r in sf_rank["ranks_list"] if r is not None]
    if len(ranks) > 0:
        summary["sf_best"]["avg_rank_present"] = float(np.mean(ranks))
        summary["sf_best"]["median_rank_present"] = float(np.median(ranks))
    else:
        summary["sf_best"]["avg_rank_present"] = None
        summary["sf_best"]["median_rank_present"] = None

    print("\n=== Stockfish top-1 in model ranking ===")
    # prints requested in italian form (Top1, Top2-3 (in aggiunta a Top1), Top4-5 (in aggiunta a Top3))
    print(f"Top1: {top1} ({100.0*top1/N:.2f}%)")
    print(f"Top2/3 (in aggiunta a Top1): {top3-top1} ({100.0*(top3-top1)/N:.2f}%)  -> Top3 complessiva: {top3} ({100.0*top3/N:.2f}%)")
    print(f"Top4/5 (in aggiunta a Top3): {top5-top3} ({100.0*(top5-top3)/N:.2f}%) -> Top5 complessiva: {top5} ({100.0*top5/N:.2f}%)")
    print(f"not in model at all: {not_in} ({100.0*not_in/N:.2f}%)")
    print(f"examples with no SF moves recorded: {no_sf} ({100.0*no_sf/N:.2f}%)")
    if len(ranks) > 0:
        print(f"Avg rank (considering only examples where SF-best in model list): {summary['sf_best']['avg_rank_present']:.3f}, median: {summary['sf_best']['median_rank_present']:.3f}")
    else:
        print("Nessun rank presente per calcolare media/mediana.")

    # salva JSON di riepilogo nella cartella out_dir
    out_json = os.path.join(out_dir, "summary.json")
    with open(out_json, "w", encoding="utf-8") as jf:
        json.dump(summary, jf, indent=2)

    print("\nTutti i grafici e i file di riepilogo sono stati salvati nella cartella:", out_dir)
    print("Summary JSON salvato in:", out_json)

    with open(os.path.join(out_dir, f"Stockfish top-1 in model ranking.txt"), "w", encoding="utf-8") as f:
        f.write("\n=== Stockfish top-1 in model ranking ===\n")
        # prints requested in italian form (Top1, Top2-3 (in aggiunta a Top1), Top4-5 (in aggiunta a Top3))
        f.write(f"Top1: {top1} ({100.0*top1/N:.2f}%)\n")
        f.write(f"Top2/3 (in aggiunta a Top1): {top3-top1} ({100.0*(top3-top1)/N:.2f}%)  -> Top3 complessiva: {top3} ({100.0*top3/N:.2f}%)\n")
        f.write(f"Top4/5 (in aggiunta a Top3): {top5-top3} ({100.0*(top5-top3)/N:.2f}%) -> Top5 complessiva: {top5} ({100.0*top5/N:.2f}%)\n")
        f.write(f"not in model at all: {not_in} ({100.0*not_in/N:.2f}%)\n")
        f.write(f"examples with no SF moves recorded: {no_sf} ({100.0*no_sf/N:.2f}%)\n")
        if len(ranks) > 0:
            f.write(f"Avg rank (considering only examples where SF-best in model list): {summary['sf_best']['avg_rank_present']:.3f}, median: {summary['sf_best']['median_rank_present']:.3f}\n")
        else:
            f.write("Nessun rank presente per calcolare media/mediana.\n")

if __name__ == "__main__":
    main()
