#!/usr/bin/env python3
"""
extract_positions_from_pgnzst.py

Estrae posizioni (FEN) + metadata + mossa UCI da un archivio PGN .zst (Lichess).
Salva output in CSV.

Usage example:
    python extract_positions_from_pgnzst.py --pgn_zst lichess_db_standard_rated_2014-07.pgn.zst --out 1Mpositions_from1Mgames_jul2014.csv --max_games 1000000 --max_positions 1000000 --sample_every 11

Options of interest:
    --out: file CSV di output
    --max_games: stop dopo questo numero di partite (None = tutte)
    --max_positions: stop dopo questo numero di righe estratte (None = tutte)
    --sample_every: prendi solo 1 posizione ogni N mosse (default 1 = tutte le mosse)
    --min_elo: filtra partite dove entrambi i giocatori >= min_elo (default 0)
"""
import argparse, io
from tqdm import tqdm
import chess.pgn
import chess
import csv
import zstandard as zstd

def open_pgn_stream(pgn_zst_path=None, from_stdin=False):
    """
    Restituisce un file-like che contiene testo PGN (decompressed).
    Usa zstandard se disponibile, altrimenti richiede che l'input sia piping da zstdcat.
    """
    if pgn_zst_path is None:
        raise ValueError("Devi fornire --pgn_zst")
    
    fh = open(pgn_zst_path, "rb")
    dctx = zstd.ZstdDecompressor()
    stream = dctx.stream_reader(fh)
    # stream è binario; chess.pgn.read_game richiede text mode -> wrap in TextIO
    return io.TextIOWrapper(stream, encoding="utf-8", errors="replace")

def result_to_value(result_tag, side_to_move_is_white):
    """
    Converte Result tag ("1-0","0-1","1/2-1/2") nel valore dal punto di vista del side_to_move
    (1.0 = win per side_to_move, 0.5 = draw, 0.0 = loss)
    """
    if result_tag == "1-0":
        winner_is_white = True
    elif result_tag == "0-1":
        winner_is_white = False
    elif result_tag in ("1/2-1/2", "1/2", "1/2-1/2\r", "1/2-1/2\n"):
        return 0.5
    else:
        # se tag mancante o altro, consideriamo 0.5 (neutral) oppure None
        return 0.5
    # se side_to_move_is_white True => vittoria se winner_is_white True
    return 1.0 if (side_to_move_is_white == winner_is_white) else 0.0

def sanitize_elo(tag):
    try:
        return int(tag)
    except Exception:
        return None

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pgn_zst", required=True, help="path .pgn.zst")
    p.add_argument("--out", "-o", required=True, help="CSV output path")
    p.add_argument("--max_games", type=int, default=None, help="max number of games da processare")
    p.add_argument("--max_positions", type=int, default=None, help="max number of positions/esempi da estrarre (globale)")
    p.add_argument("--sample_every", type=int, default=1, help="prendi solo 1 posizione ogni N mosse (1 = tutte)")
    p.add_argument("--min_elo", type=int, default=0, help="filtra partite con entrambi i giocatori >= min_elo")
    args = p.parse_args()

    src = open_pgn_stream(args.pgn_zst)
    # apri csv
    out_fh = open(args.out, "w", encoding="utf-8", newline="")
    writer = csv.writer(out_fh)
    header = ["game_id","white","black","white_elo","black_elo","result_tag",
              "ply","fen_before","side_to_move_is_white","elo_side","move_uci","move_san","y_value"]
    writer.writerow(header)

    games_proc = 0
    positions_written = 0
    # chess.pgn.read_game vuole un TextIO
    pgn_io = src #if args.from_pgn_stream else src
    # if it's binary TextIOWrapper we can use it directly
    # loop through games
    pbar_games = tqdm(total=args.max_games, desc="games", disable=False) if args.max_games else None
    while True:
        try:
            game = chess.pgn.read_game(pgn_io)
        except Exception as e:
            print("Errore lettura PGN:", e)
            break
        if game is None:
            break
        games_proc += 1
        if pbar_games:
            pbar_games.update(1)
        # read headers
        white = game.headers.get("White", "")
        black = game.headers.get("Black", "")
        white_elo = sanitize_elo(game.headers.get("WhiteElo", ""))
        black_elo = sanitize_elo(game.headers.get("BlackElo", ""))
        result_tag = game.headers.get("Result", "")
        
        if white_elo is not None and black_elo is not None:
            if (white_elo < args.min_elo) or (black_elo < args.min_elo):
                # skip partita
                pass_flag = True
            else:
                pass_flag = False
        else:
            pass_flag = False

        if pass_flag:
            # skip this game
            if args.max_games and games_proc >= args.max_games:
                break
            if args.max_games is None:
                continue
            else:
                continue

        # iterate moves
        board = game.board()  # initial board
        node = game
        ply = 0
        while node.variations:
            move = node.variations[0].move
            # fen before the move
            fen_before = board.fen()
            side_to_move_is_white = board.turn  # True if white to move

            # check legality (python-chess ensures move is legal if parsed from PGN)
            # but be cautious: some PGNs contain comments/annotations; we ensure move in legal_moves
            if move not in board.legal_moves:
                # Skip if move not legal for current board (data inconsistency)
                node = node.variations[0]
                board.push(move)  # still advance to keep sync (or consider break)
                ply += 1
                continue

            # compute y_value (from point of view of side_to_move BEFORE the move)
            y_val = result_to_value(result_tag, side_to_move_is_white)

            # if sampling (sample_every), decide whether to output
            if (ply % args.sample_every) == 0:
                # choose elo of side to move (normalized later)
                elo_side = white_elo if side_to_move_is_white else black_elo
                # safe default if elos missing
                elo_side_norm = elo_side if elo_side is not None else 0

                # move UCI and SAN
                uci = move.uci()
                try:
                    san = board.san(move)
                except Exception:
                    san = "?"

                # write CSV row
                writer.writerow([games_proc, white, black, white_elo or "", black_elo or "",
                                 result_tag, ply, fen_before, int(side_to_move_is_white),
                                 elo_side_norm, uci, san, y_val])
                positions_written += 1

                if args.max_positions and positions_written >= args.max_positions:
                    break

            # advance
            board.push(move)
            node = node.variations[0]
            ply += 1

        if args.max_positions and positions_written >= args.max_positions:
            break
        if args.max_games and games_proc >= args.max_games:
            break

    if pbar_games:
        pbar_games.close()
    out_fh.close()
    print(f"Fatto. Games processati: {games_proc}, posizioni scritte: {positions_written}")

if __name__ == "__main__":
    main()
