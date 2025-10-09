#!/usr/bin/env python3
"""
validate_npz_dataset.py

Controllo minimo su file .npz con X_boards, X_eloside, y, y_value, legal_indices

Usage example:
    python validate_npz_dataset.py first_dataset_100k.npz move2idx_generated.json --check_n 100000
"""

import json, argparse, numpy as np
from tqdm import trange
import chess
import collections

def planes_to_board(planes, side_to_move, halfmove_clock, number_move):
    board = chess.Board()
    board.clear()
    piece_type_map = {0: chess.PAWN, 1: chess.ROOK, 2: chess.KNIGHT,
                      3: chess.BISHOP, 4: chess.QUEEN, 5: chess.KING}
    for r in range(8):
        for f in range(8):
            for p in range(13):
                if planes[r, f, p]:
                    sq_rank = 7 - r
                    sq_file = f
                    sq = chess.square(sq_file, sq_rank)
                    
                    # Se siamo sul piano 12 e 
                    if p == 12:
                        if r != 0 and r != 7:
                            board.ep_square = sq
                    else:
                        piece_type = piece_type_map[p % 6]
                        color = chess.WHITE if p < 6 else chess.BLACK
                        board.set_piece_at(sq, chess.Piece(piece_type, color))

    board.turn = True if side_to_move == 1.0 else False

    castle = ''.join([int(planes[0,0,12])*'K', int(planes[0,7,12])*'Q', int(planes[7,0,12])*'k', int(planes[7,7,12])*'q'])
    board.set_castling_fen(castle)

    board.halfmove_clock = int(halfmove_clock)
    board.fullmove_number = int(number_move)

    return board

def load_npz_dataset(npz_path):
    print("Carico dataset:", npz_path)
    data = np.load(npz_path, allow_pickle=True)
    X_boards = data["X_boards"]   # (N,8,8,13)
    X_eloside = data["X_eloside"] # (N,5)
    y = data["y"]                 # (N,)
    y_value = data["y_value"]     # (N,)
    legal_indices = data.get("legal_indices", None) # (N,num_classes)

    print("Shapes:", X_boards.shape, X_eloside.shape, y.shape, y_value.shape, legal_indices.shape)
    print("Type:", X_boards.dtype, X_eloside.dtype, y.dtype, y_value.dtype, legal_indices.dtype)

    num_classes = len(legal_indices[0])
    print("Numero classi (mosse) =", num_classes)
    return X_boards, X_eloside, y, y_value, num_classes, legal_indices 

def main():
    p = argparse.ArgumentParser()
    p.add_argument("npz", help="dataset .npz (es. dataset_with_result_20k_from_fen.npz)")
    p.add_argument("move2idx", help="move2idx json (es. move2idx_generated.json)")
    p.add_argument("--check_n", type=int, default=2000, help="numero di esempi da controllare random (default 2000)")
    args = p.parse_args()

    X_boards, X_eloside, y, y_value, num_classes, legal_indices = load_npz_dataset(args.npz)
    N = X_boards.shape[0]

    with open(args.move2idx, "r", encoding="utf-8") as f:
        move2idx = json.load(f)
    idx2move = {int(v): k for k, v in move2idx.items()}
    if num_classes == len(idx2move):
        print("Numero classi = Numero vocabolario")
    else:
        print(f"Numero classi {num_classes} != Numero vocabolario {len(idx2move)}")

    # check basic ranges
    print("\nControlli base:")
    print("  - side_to_move sample values (unique):", np.unique(X_eloside[:,2])[:10])
    print("  - white elo (min,max):", round(float(np.min(X_eloside[:,0]))*1000+1000,0), round(float(np.max(X_eloside[:,0]))*1000+1000,0))
    print("  - black elo (min,max):", round(float(np.min(X_eloside[:,1]))*1000+1000,0), round(float(np.max(X_eloside[:,1]))*1000+1000,0))
    print("  - y_value unique (sample):", np.unique(y_value)[:10])

    # Sampling indices to check legality
    rng = np.random.RandomState()
    check_n = min(args.check_n, N)
    idxs = rng.choice(N, size=check_n, replace=False)

    illegal_count = 0
    side_mismatch = 0
    missing_move_in_vocab = 0
    for i in trange(check_n, desc="validating samples"):
        planes = X_boards[idxs[i]]
        eloside = X_eloside[idxs[i]]
        yi = int(y[idxs[i]])
        # reconstruct board
        try:
            board = planes_to_board(planes, eloside[2], eloside[3], eloside[4])
        except Exception:
            illegal_count += 1
            continue

        # check side consistency
        side_flag = 1.0 if board.turn == chess.WHITE else 0.0
        if abs(side_flag - eloside[2]) > 0.001:
            side_mismatch += 1

        # get true move uci from idx2move
        true_uci = idx2move.get(yi, None)
        if true_uci is None:
            missing_move_in_vocab += 1
            continue
        try:
            mv = chess.Move.from_uci(true_uci)
        except Exception:
            missing_move_in_vocab += 1
            continue

        # is legal?
        if mv not in board.legal_moves:
            illegal_count += 1

    print("\n=== VALIDATION SUMMARY ===")
    print(f"Samples checked: {check_n} / {N}")
    print(f"  - true move not in vocab: {missing_move_in_vocab}")
    print(f"  - samples where true move NOT legal in reconstructed board: {illegal_count} "
          f"({illegal_count/check_n*100:.2f}%)")
    print(f"  - side_to_move mismatches (board.turn vs X_eloside[:,2]): {side_mismatch}  "
          f"({side_mismatch/check_n*100:.2f}%)")
    # extra statistics
    print("\nDistribuzioni (y_value):")
    cnt = collections.Counter(y_value.tolist())
    for k,v in sorted(cnt.items()):
        print(f"  y_value={k}: {v} samples (overall)")

    # alcuni check rapidi sui valori y (range)
    if np.any(y < 0) or np.any(y >= len(idx2move)):
        print("ATTENZIONE: esistono indici y fuori dal range della mappa move2idx!")
    else:
        print(f"OK: tutti gli indici y sono dentro il range [0, {len(idx2move)}).")

if __name__ == '__main__':
    main()
