#!/usr/bin/env python3
"""
lets_try.py

Test for chess_elo_model:

Options of interest:
    
    --fen: FEN string (per provare una posizione arbitraria)
    --elo_w: white_elo
    --elo_b: black_elo
    
    --dataset: dataset .npz (per usare --index o --export_csv)
    --index: indice nell'npz (0-based)
    
    --export_csv: path csv per esportare predizioni (richiede --npz)
    --start: start index per export (inclusive)
    --end: end index per export (exclusive) [don't use if all]

    --illegal: 0 = eliminare mosse illegali consigliate
    --topk: top-k moves consigliate

    Usage example:
        python lets_try.py --model chess_elo_model_V1.keras --move2idx move2idx_all.json --topk 5 --dataset validation_10k_positions_from_130_files.npz --index 100
        python lets_try.py --model chess_elo_model_V1.keras --move2idx move2idx_all.json --topk 5 --dataset validation_10k_positions_from_130_files.npz --export_csv 100Examples.csv --start 0 --end 101
        python lets_try.py --model chess_elo_model_V1.keras --move2idx move2idx_all.json --topk 5 --fen "5k2/1p6/2p1b2p/p3PppN/4p1P1/1P5P/P3KP2/8 w - f6 0 34" --elo_w 1399 --elo_b 1666

"""
import argparse, json, csv
import numpy as np
import tensorflow as tf
import chess
from stockfish import Stockfish
from tqdm import tqdm

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

def board_from_fen_to_planes(fen):
    board = chess.Board(fen)
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

    side_to_move = 1.0 if board.turn == chess.WHITE else 0.0

    return planes, halfmove_clock, moves, side_to_move

def mask_and_topk(p, legal_moves_indices, idx2move, topk=10, illegal=1):
    
    list_moves = []
    for idx, prob in enumerate(p):
        if illegal == 0:
            if legal_moves_indices[idx] == 1:
                uci = idx2move.get(int(idx), None)
                list_moves.append((uci, float(prob)))
        else:
            uci = idx2move.get(int(idx), None)
            list_moves.append((uci, float(prob)))
            
    probs = np.array([it[1] for it in list_moves], dtype=np.float32)
    s = probs.sum()
    probs = probs / s
    items = sorted(zip([it[0] for it in list_moves], probs), key=lambda x: x[1], reverse=True)

    return items[:topk]

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

def export_csv(model, idx2move, Xb, Xe, y, yv, legal_indices, out_csv, topk, illegal, start=0, end=-1, batch_size=128):

    stockfish = Stockfish("C:\\Users\\Andrea\\Documents\\GitHubRepo\\stockfish\\stockfish-windows-x86-64-avx2.exe")

    N = Xb.shape[0]
    if end == -1 or end > N:
        end = N
    if start < 0 or start >= N:
        raise ValueError("start out of range")
    indices = list(range(start, end))
    M = len(indices)
    print(f"Exporting {M} samples to {out_csv} (indices {start}..{end-1})")

    # predizioni in batch
    preds_policy = []
    preds_value = []
    B = batch_size
    for i in range(0, M, B):
        sel = indices[i:i+B]
        xb = Xb[sel].astype(np.float32)
        xe = Xe[sel].astype(np.float32)
        xi = legal_indices[sel].astype(np.float32)
        p_batch, v_batch = model.predict({"board": xb, "eloside": xe, "legal_indices": xi}, verbose=0)
        preds_policy.append(p_batch)
        preds_value.append(v_batch[:,0])
    preds_policy = np.vstack(preds_policy)
    preds_value = np.concatenate(preds_value)

    # write CSV
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        header = ["index","fen","side","elo_side","elo_opponent","true_move","true_result","value_pred","true_move_prob","topk","topk_prob","topk_stockfish","legal_moves"]
        writer.writerow(header)
        pbar = tqdm(total=len(indices), desc="positions", disable=False)
        for i_idx, idx in enumerate(indices):
            planes = Xb[idx]
            eloside = Xe[idx]
            legal_example_indices = legal_indices[idx]
            true_idx = int(y[idx])
            true_move = idx2move.get(true_idx, "<unk>")
            true_val = float(yv[idx])
            if true_val >= 0.99:
                true_res = "WIN"
            elif true_val >= 0.49:
                true_res = "DRAW"
            else:
                true_res = "LOSS"
            board = planes_to_board(planes, eloside[2], eloside[3], eloside[4])
            stockfish.set_fen_position(board.fen())
            sto_topk = stockfish.get_top_moves(topk)
            moss_topk = []
            for i in range(len(sto_topk)):
                try:
                    mv = chess.Move.from_uci(sto_topk[i]["Move"])
                    moss_topk.append(f"{board.san(mv)}")
                except Exception:
                    moss_topk.append(sto_topk[i]["Move"])
            try:
                true_move_san = board.san(chess.Move.from_uci(true_move))
            except Exception:
                true_move_san = "?"
            p = preds_policy[i_idx]
            v = float(preds_value[i_idx])
            top = mask_and_topk(p, legal_example_indices, idx2move, topk=topk, illegal=illegal)
            # prob assigned to true move (raw)
            tprob = float(p[true_idx]) if true_idx < p.shape[0] else 0.0
            # format topk as uci:prob%
            top_san = top
            i = 0
            for (u, prob) in top:
                try:
                    top_san[i] = (board.san(chess.Move.from_uci(u)), prob)
                except Exception:
                    top_san[i] = ("?", prob)
                i += 1

            side_str = "White" if eloside[2]==1.0 else "Black"
            elo_side = (eloside[0] if eloside[2]==1.0 else eloside[1])*1000 + 1000
            elo_opponent = (eloside[0] if eloside[2]==0.0 else eloside[1])*1000 + 1000

            topk_str = ", ".join([f"{u}" for (u,prob) in top_san if prob != 0])
            topk_str_prob = ", ".join([f"{prob*100:.2f}%" for (u,prob) in top_san if prob != 0])
            moss_topk = ", ".join([f"{m}" for m in moss_topk])
            legal_moves = ", ".join([f"{board.san(m)}" for m in board.legal_moves])

            writer.writerow([idx, board.fen(), side_str, f"{elo_side:.0f}", f"{elo_opponent:.0f}", true_move_san, true_res, f"{v:.4f}", f"{tprob*100:.2f}%", topk_str, topk_str_prob, moss_topk, legal_moves])
            pbar.update(1)

        pbar.close()

    print("Export completato.")

def infer_from_index(model, idx2move, X_boards, X_eloside, y, y_value, legal_indices, index, topk, illegal):
    
    stockfish = Stockfish("C:\\Users\\Andrea\\Documents\\GitHubRepo\\stockfish\\stockfish-windows-x86-64-avx2.exe")

    if index < 0 or index >= X_boards.shape[0]:
        print("Index out of range:", index)
        return

    planes = X_boards[index].astype(np.float32)
    eloside = X_eloside[index].astype(np.float32)
    legal_example_indices = legal_indices[index].astype(np.float32)
    true_idx = int(y[index])
    true_move_uci = idx2move.get(true_idx, "<unk>")
    true_val = float(y_value[index])
    if true_val >= 0.99:
        true_res = "WIN"
    elif true_val >= 0.49:
        true_res = "DRAW"
    else:
        true_res = "LOSS"

    board = planes_to_board(planes, eloside[2], eloside[3], eloside[4])

    stockfish.set_fen_position(board.fen())
    sto_topk = stockfish.get_top_moves(topk)
    moss_topk = []
    for i in range(len(sto_topk)):
        try:
            mv = chess.Move.from_uci(sto_topk[i]["Move"])
            moss_topk.append(board.san(mv))
        except Exception:
            moss_topk.append(sto_topk[i]["Move"])

    # predizione del modello
    b = np.expand_dims(planes, axis=0)
    e = np.expand_dims(eloside, axis=0)
    i = np.expand_dims(legal_example_indices, axis=0)
    p, v = model.predict({"board": b, "eloside": e, "legal_indices": i}, verbose=0)
    p = p[0]
    v = float(v[0,0])

    # topk legali dal modello
    top = mask_and_topk(p, legal_example_indices, idx2move, topk=topk, illegal=illegal)

    prob_assigned = 0.0
    true_in_topk = False
    true_rank = 1
    for move in top:
        if move[0] == true_move_uci:
            prob_assigned = float(move[1])
            true_in_topk = True
            break
        else:
            true_rank += 1
    
    if true_rank >= topk:
        true_rank = None

    if prob_assigned == 0.0:
        for idx, prob in enumerate(p):
            if idx2move[idx] == true_move_uci:
                prob_assigned = float(prob)
                print("La probabilità assegnata dal modello alla true move sarà non riaggiornata rispetto alle mosse legali")
                break

    # stampa informazioni
    print(f"Index {index}  Side to move: {'White' if eloside[2]==1.0 else 'Black'}  elo(side) ~ { (eloside[0] if eloside[2]==1.0 else eloside[1])*1000+1000 :.0f}")
    print(f"Value predetto: {v:.4f} -> {v*100:.2f}%")
    print(f"True move in data: UCI = {true_move_uci}   result = {true_res} (y_value={true_val})")
    # prova a stampare la SAN della mossa vera se possibile
    try:
        mv_true = chess.Move.from_uci(true_move_uci)
        san_true = board.san(mv_true) if mv_true in board.legal_moves else "(SAN non disponibile: mossa non legale nella board ricostruita)"
        print(f"True move SAN (se calcolabile): {san_true}")
    except Exception:
        print("True move SAN: (impossibile convertire)")

    print(f"Probabilità assegnata dal modello alla true move (raw): {prob_assigned:.6f} -> {prob_assigned*100:.2f}%")
    if true_in_topk:
        print(f"La true move è presente nelle top-{topk} predette (rank={true_rank}).")
    else:
        print(f"La true move NON è presente nelle top-{topk} predette.")

    # stampa top-k legali
    print("Top candidate legali (SAN  UCI  prob):")
    for uci, prob in top:
        try:
            mv = chess.Move.from_uci(uci)
            san = board.san(mv)
        except Exception:
            san = "?"
        print(f"  {san:6}   {uci:6}   {prob*100:5.2f}%")

    print("\nBoard ascii:")
    print(board)
    print("Board fen:")
    print(board.fen())
    print("Board legal moves:")
    print(board.legal_moves)
    print("Mosse migliori Stockfish:")
    print(moss_topk)

def infer_from_fen(model, move2idx, idx2move, fen, elo_w, elo_b, topk, illegal):

    stockfish = Stockfish("C:\\Users\\Andrea\\Documents\\GitHubRepo\\stockfish\\stockfish-windows-x86-64-avx2.exe")

    white_elo = float(elo_w)
    black_elo = float(elo_b)
    mean_elo = (white_elo + black_elo) / 2
    w_elo_norm = (white_elo - 1000.0) / 1000.0
    b_elo_norm = (black_elo - 1000.0) / 1000.0

    planes, halfmove_clock, number_move, side_to_move = board_from_fen_to_planes(fen)
    
    stockfish.set_fen_position(fen)
    sto_topk = stockfish.get_top_moves(topk)
    moss_topk = []
    for i in range(len(sto_topk)):
        try:
            mv = chess.Move.from_uci(sto_topk[i]["Move"])
            moss_topk.append(board.san(mv))
        except Exception:
            moss_topk.append(sto_topk[i]["Move"])

    eloside = []
    eloside.append([w_elo_norm, b_elo_norm, side_to_move, halfmove_clock, number_move])
    eloside = np.array(eloside, dtype=np.float32)
    eloside = eloside[0]

    board = planes_to_board(planes, eloside[2], eloside[3], eloside[4])

    legal_example = board.legal_moves
    legal_example_indices = [0] * len(move2idx)
    for m in legal_example:
        u = m.uci()
        if u in move2idx:
            legal_example_indices[move2idx[u]] = 1
    
    planes = planes.astype(np.float32)
    legal_example_indices = np.array(legal_example_indices, dtype=np.float32)

    b = np.expand_dims(planes, axis=0)
    e = np.expand_dims(eloside, axis=0)
    i = np.expand_dims(legal_example_indices, axis=0)
    p, v = model.predict({"board": b, "eloside": e, "legal_indices": i}, verbose=0)
    p = p[0]
    v = float(v[0,0])

    # topk legali dal modello
    top = mask_and_topk(p, legal_example_indices, idx2move, topk=topk, illegal=illegal)

    # stampa informazioni
    print(f"Side to move: {'White' if eloside[2]==1.0 else 'Black'}  elo(side) ~ { (eloside[0] if eloside[2]==1.0 else eloside[1])*1000+1000 :.0f}")
    print(f"Value predetto: {v:.4f} -> {v*100:.2f}%")
    # stampa top-k legali
    print("Top candidate legali (SAN  UCI  prob):")
    for uci, prob in top:
        try:
            mv = chess.Move.from_uci(uci)
            san = board.san(mv)
        except Exception:
            san = "?"
        print(f"  {san:6}   {uci:6}   {prob*100:5.2f}%")

    print("Board ascii:")
    print(board)
    print("Board fen:")
    print(board.fen())
    print("Board legal moves:")
    print(board.legal_moves)
    print("Mosse migliori Stockfish:")
    print(moss_topk)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--move2idx", required=True)
    p.add_argument("--dataset", help="dataset .npz (per usare --index o --export_csv)")
    p.add_argument("--index", type=int, help="index nell'npz (0-based)")
    p.add_argument("--fen", help="FEN string (se vuoi provare una posizione arbitraria)")
    p.add_argument("--elo_w", help="white_elo")
    p.add_argument("--elo_b", help="black_elo")
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--export_csv", help="path csv per esportare predizioni (richiede --npz)")
    p.add_argument("--start", default=0,type=int, help="start index per export (inclusive)")
    p.add_argument("--end", default=-1, type=int, help="end index per export (exclusive)")
    p.add_argument("--illegal", default=1, type=int, help="0 per eliminare le mosse illegali cosigliate")
    args = p.parse_args()

    print("Carico modello:", args.model)
    model = tf.keras.models.load_model(args.model)

    with open(args.move2idx, "r", encoding="utf-8") as f:
        move2idx = json.load(f)
    idx2move = {int(v): k for k, v in move2idx.items()}

    if args.dataset:
        X_boards, X_eloside, y, y_value, num_classes, legal_indices = load_npz_dataset(args.dataset)

    if args.export_csv:
        export_csv(model, idx2move, X_boards, X_eloside, y, y_value, legal_indices, args.export_csv, args.topk, args.illegal, args.start, args.end, batch_size=128)
    
    if args.index:
        infer_from_index(model, idx2move, X_boards, X_eloside, y, y_value, legal_indices, int(args.index), args.topk, args.illegal)

    if args.fen:
        infer_from_fen(model, move2idx, idx2move, args.fen, args.elo_w, args.elo_b, args.topk, args.illegal)