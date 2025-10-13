import chess
import json

moves = []

# Queen
for x in range(8):
	for y in range(8):
		board = chess.Board("8/8/8/8/8/8/8/8")
		board.set_piece_at(chess.square(x,y), chess.Piece(chess.QUEEN, chess.WHITE))
		moves += [move.uci() for move in board.generate_legal_moves()]

# Knight
for x in range(8):
	for y in range(8):
		board = chess.Board("8/8/8/8/8/8/8/8")
		board.set_piece_at(chess.square(x,y), chess.Piece(chess.KNIGHT, chess.WHITE))
		moves += [move.uci() for move in board.generate_legal_moves()]


# Pawn promotions, 2 players, 8 ranks each, 4 choices (queen, rook, bishop, knight)
# Diagonal pawn promotions, 2 players, 14 diagonals (12 in the center and 1 each in the first and last ranks)
for x in range(8):
	board = chess.Board("8/8/8/8/8/8/8/8")
	board.set_piece_at(chess.square(x,6), chess.Piece(chess.PAWN, chess.WHITE))
	if x > 0:
		board.set_piece_at(chess.square(x-1,7), chess.Piece(chess.PAWN, chess.BLACK))
	if x < 7:
		board.set_piece_at(chess.square(x+1,7), chess.Piece(chess.PAWN, chess.BLACK))
	moves += [move.uci() for move in board.generate_legal_moves()]

	board = chess.Board("8/8/8/8/8/8/8/8")
	board.turn = chess.BLACK
	board.set_piece_at(chess.square(x,1), chess.Piece(chess.PAWN, chess.BLACK))
	if x > 0:
		board.set_piece_at(chess.square(x-1,0), chess.Piece(chess.PAWN, chess.WHITE))
	if x < 7:
		board.set_piece_at(chess.square(x+1,0), chess.Piece(chess.PAWN, chess.WHITE))
	moves += [move.uci() for move in board.generate_legal_moves()]
	board.turn = chess.WHITE


print(sorted(moves))
print(len(moves))
assert len(moves) == len(set(moves))

move2idx = {move: idx for idx,move in enumerate(moves)}
with open("move2idx_all.json","w",encoding="utf-8") as f:
	json.dump(move2idx,f,indent=2)
print(f"  -> mosse contate: {len(moves)}, top_k={len(move2idx)} salvato in move2idx_all.json")