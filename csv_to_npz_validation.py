#!/usr/bin/env python3
"""
Legge CSV con colonne minime:
  fen_before, move_uci, y_value, white_elo, black_elo
  
  Costruisce:
  - dataset .npz con X_boards, X_eloside, y, y_value, legal_indices

Usage example:
    python csv_to_npz_validation.py all_positions_jul2014_csv/positions_jul2014 validation_100k_positions_from_130_files.npz --json move2idz_all.json --selected_indices_out validation_selected_indices.json --example_from_file 769 --game_split 1500 --max_games 1048440 --max_file 130
"""

import argparse, json
from collections import Counter
import numpy as np
import pandas as pd
import chess
import os
import random

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

def create_dataset_from_csv(X_boards, X_eloside, y, y_value, legal_indices, csv_path, move2idx, indices_size, max_examples):
    examples = 0

    with open(csv_path, "r", encoding="utf-8", errors="ignore") as fh:
        total_lines = sum(1 for _ in fh)

    num_rows = total_lines - 1 # tolgo header

    selected_indices = set(random.sample(range(num_rows), max_examples))

    # pandas chunk read
    global_row = 0
    for chunk in pd.read_csv(csv_path, chunksize=2000):
        for _, row in chunk.iterrows():
            if global_row in selected_indices:
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

                if examples >= max_examples:
                    break
            
            global_row += 1

        if examples >= max_examples:
            break

    if examples == 0:
        raise RuntimeError("Nessun esempio creato: controlla il CSV e il vocabolario move2idx")

    return X_boards, X_eloside, y, y_value, legal_indices, selected_indices

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_in", help="CSV input")
    parser.add_argument("out_npz", help="Output .npz")
    parser.add_argument("--json", help="vocabulary")
    parser.add_argument("--example_from_file", type=int, default=10, help="totale esempi salvati da ogni file")
    parser.add_argument("--game_split", type=int, default=None, help="number of games in ogni file in input")
    parser.add_argument("--max_games", type=int, default=None, help="max number of games in totale")
    parser.add_argument("--max_file", type=int, default=1000, help="Numero massimo di file da processare")
    parser.add_argument("--start", type=int, default=1, help="File number to start")
    parser.add_argument("--selected_indices_out")
    args = parser.parse_args()

    with open(args.json, "r", encoding="utf-8") as f:
        move2idx = json.load(f)

    base_in = os.path.splitext(args.csv_in)[0]  # es: all_positions_jul2014_csv/positions_jul2014
    base_out = os.path.splitext(args.out_npz)[0]  # es: all_positions_jul2014_npz/positions_jul2014_npz

    X_boards = []
    X_eloside = []
    y = []
    y_value = []
    legal_indices = []

    indices = {}
    start = 1 + (args.start - 1) * args.game_split
    batch = 1
    while batch <= args.max_file:
        end = min(start + args.game_split - 1, args.max_games)
        in_file = f"{base_in}_game{start}_game{end}.csv"

        print(f"[Batch {batch}] extracting {args.example_from_file} positions from {in_file}")

        print("Creo esempi (posizione->mossa) dalla FEN (seconda passata)...")
        X_boards, X_eloside, y, y_value, legal_indices, selected_indices = create_dataset_from_csv(X_boards, X_eloside, y, y_value, legal_indices, in_file, move2idx, indices_size=len(move2idx), max_examples=args.example_from_file)

        indices[in_file] = tuple(selected_indices)
        start += args.game_split
        batch += 1

    with open(args.selected_indices_out,"w",encoding="utf-8") as f:
        json.dump(indices,f,indent=2)

    X_boards = np.array(X_boards, dtype=np.uint8)
    X_eloside = np.array(X_eloside, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    y_value = np.array(y_value, dtype=np.float32)
    legal_indices = np.array(legal_indices, dtype=np.uint8)
    print(f"Esempi creati: {len(y)}")
    print("Salvo .npz compresso...")
    np.savez_compressed(args.out_npz, X_boards=X_boards, X_eloside=X_eloside, y=y, y_value=y_value, legal_indices=legal_indices)
    print(f"Salvato {args.out_npz}")

if __name__ == "__main__":
    main()
