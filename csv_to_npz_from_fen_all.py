#!/usr/bin/env python3
"""
csv_to_npz_from_fen.py

Legge CSV con colonne minime:
  fen_before, move_uci, y_value, white_elo, black_elo
  
  Costruisce:
  - move2idx.json
  - dataset .npz con X_boards, X_eloside, y, y_value, legal_indices

Usage example:
    python csv_to_npz_from_fen_all.py all_positions_jul2014_csv/positions_jul2014.csv all_positions_jul2014_npz/positions_jul2014.npz --voc_file positions_for_vocabulary_jul2014.csv --max_games 1048440 --game_split 1500 --max_file 699
"""

import argparse, json
from collections import Counter
import numpy as np
import pandas as pd
import chess
import os

def board_to_planes(board):
    piece_map = board.piece_map()
    planes = np.zeros((8,8,13), dtype=np.uint8)
    piece_to_plane = {
        chess.PAWN: 0, chess.ROOK: 1, chess.KNIGHT: 2,
        chess.BISHOP: 3, chess.QUEEN: 4, chess.KING: 5
    }
    # planes[:,:,12] for en passant / arrocchi
    if board.has_legal_en_passant():
        planes[7 - chess.square_rank(board.ep_square), chess.square_file(board.ep_square), 12] = 1
    if board.has_kingside_castling_rights(chess.WHITE):
        planes[0,0,12] = 1
    if board.has_queenside_castling_rights(chess.WHITE):
        planes[0,7,12] = 1
    if board.has_kingside_castling_rights(chess.BLACK):
        planes[7,0,12] = 1
    if board.has_queenside_castling_rights(chess.BLACK):
        planes[7,7,12] = 1

    halfmove_clock = board.halfmove_clock
    moves = board.fullmove_number

    for sq, piece in piece_map.items():
        rank = 7 - chess.square_rank(sq)
        file = chess.square_file(sq)
        base = piece_to_plane[piece.piece_type]
        plane = base if piece.color == chess.WHITE else base + 6
        planes[rank, file, plane] = 1
    return planes, halfmove_clock, moves

def build_move_vocab(csv_path, max_rows=None):
    cnt = Counter()
    read = 0
    for chunk in pd.read_csv(csv_path, chunksize=10000):
        for _, row in chunk.iterrows():
            if max_rows and read >= max_rows:
                break
            uci = row.get("move_uci", None)
            if isinstance(uci, str) and uci.strip() != "":
                cnt[uci] += 1
            read += 1
        if max_rows and read >= max_rows:
            break
    return cnt

def create_dataset_from_csv(csv_path, move2idx, indices_size, max_examples=None):
    X_boards = []
    X_eloside = []
    y = []
    y_value = []
    legal_indices = []
    examples = 0

    # pandas chunk read
    for chunk in pd.read_csv(csv_path, chunksize=2000):
        for _, row in chunk.iterrows():
            # required fields
            fen = row.get("fen_before", None)
            uci = row.get("move_uci", None)
            result_tag = row.get("y_value", 0.5)
            if fen is None or not isinstance(uci, str) or uci.strip() == "":
                continue

            try:
                board = chess.Board(fen)
            except Exception:
                # malformed fen
                continue

            legal_example = board.legal_moves

            # sanity: check that the true move is legal in this board
            try:
                true_move_is_legal = (chess.Move.from_uci(uci) in legal_example)
            except Exception:
                true_move_is_legal = False

            if not true_move_is_legal:
                # skip inconsistent rows
                continue

            # elo normalization (if missing, use 0)
            try:
                white_elo = float(row.get("white_elo", 0.0))
            except Exception:
                white_elo = 0.0
            try:
                black_elo = float(row.get("black_elo", 0.0))
            except Exception:
                black_elo = 0.0
            w_elo_norm = (white_elo - 1000.0) / 1000.0
            b_elo_norm = (black_elo - 1000.0) / 1000.0

            # build planes from fen (this includes castling/ep correctly)
            planes, halfmove_clock, number_move  = board_to_planes(board)

            # compute side_to_move (1.0 white, 0.0 black)
            side_to_move = 1.0 if board.turn == chess.WHITE else 0.0

            # only keep if uci in move2idx
            if uci in move2idx:
                val = result_tag
                X_boards.append(planes)
                X_eloside.append([w_elo_norm, b_elo_norm, side_to_move, halfmove_clock, number_move])
                y.append(move2idx[uci])
                y_value.append(val)
                legal_example_indices = [0] * indices_size
                for m in legal_example:
                    u = m.uci()
                    if u in move2idx:
                        legal_example_indices[move2idx[u]] = 1
                legal_indices.append(legal_example_indices)

                examples += 1
                if max_examples and examples >= max_examples:
                    break

        if max_examples and examples >= max_examples:
            break

    if examples == 0:
        raise RuntimeError("Nessun esempio creato: controlla il CSV e il vocabolario move2idx")

    X_boards = np.array(X_boards, dtype=np.uint8)
    X_eloside = np.array(X_eloside, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    y_value = np.array(y_value, dtype=np.float32)
    legal_indices = np.array(legal_indices, dtype=np.uint8)
    return X_boards, X_eloside, y, y_value, legal_indices

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_in", help="CSV input")
    parser.add_argument("out_npz", help="Output .npz")
    parser.add_argument("--json", default=None, help="vocabulary")
    parser.add_argument("--voc_file", help="file per move2idx")
    parser.add_argument("--max_rows_vocab", type=int, default=10000000, help="numero righe usate per costruire vocabolario")
    parser.add_argument("--top_k", type=int, default=1968, help="top K mosse per vocabolario") # max: 1968
    parser.add_argument("--max_examples", type=int, default=None, help="limite totale esempi salvati")
    parser.add_argument("--game_split", type=int, default=None, help="number of games in ogni file in input")
    parser.add_argument("--max_games", type=int, default=None, help="max number of games in totale")
    parser.add_argument("--max_file", type=int, default=1000, help="Numero massimo di file da processare")
    parser.add_argument("--start", type=int, default=1, help="File number to start")
    args = parser.parse_args()

    if args.json:
        with open(args.json, "r", encoding="utf-8") as f:
            move2idx = json.load(f)
    else:
        print("Costruisco vocabolario delle mosse (prima passata)...")
        print(f"da {args.voc_file}")
        cnt = build_move_vocab(args.voc_file, max_rows=args.max_rows_vocab)
        most_common = cnt.most_common(args.top_k)
        move2idx = {move: idx for idx,(move,_) in enumerate(most_common)}
        with open("move2idx_generated.json","w",encoding="utf-8") as f:
            json.dump(move2idx,f,indent=2)
        print(f"  -> mosse contate: {len(cnt)}, top_k={len(move2idx)} salvato in move2idx_generated.json")

    base_in = os.path.splitext(args.csv_in)[0]  # es: all_positions_jul2014_csv/positions_jul2014
    base_out = os.path.splitext(args.out_npz)[0]  # es: all_positions_jul2014_npz/positions_jul2014_npz

    start = 1 + (args.start - 1) * args.game_split
    batch = 1
    while batch <= args.max_file:
        end = min(start + args.game_split - 1, args.max_games)
        in_file = f"{base_in}_game{start}_game{end}.csv"
        out_file = f"{base_out}_game{start}_game{end}.npz"

        cmd = [
            "python",
            "csv_to_npz_from_fen.py",
            in_file,
            out_file,
        ]

        print(f"[Batch {batch}] Running: {' '.join(cmd)}")

        print("Creo esempi (posizione->mossa) dalla FEN (seconda passata)...")
        X_boards, X_eloside, y, y_value, legal_indices = create_dataset_from_csv(in_file, move2idx, indices_size=len(move2idx), max_examples=args.max_examples)
        print(f"Esempi creati: {len(y)}")
        print("Salvo .npz compresso...")
        np.savez_compressed(out_file, X_boards=X_boards, X_eloside=X_eloside, y=y, y_value=y_value, legal_indices=legal_indices)
        print(f"Salvato {out_file}")

        start += args.game_split
        batch += 1

if __name__ == "__main__":
    main()
