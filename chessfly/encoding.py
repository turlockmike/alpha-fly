"""Board -> feature vector and move <-> policy index, always from the side to move's view
(black's positions are flipped vertically with colours swapped, so the net only ever plays "up")."""

from __future__ import annotations

import chess
import numpy as np

N_BASE = 13 * 64 + 6         # 12 piece planes + en-passant plane, 4 castling rights, halfmove clock, repetition
N_FEATURES = N_BASE + 129    # + squares attacked by us, squares attacked by them, are we in check.
# The attack planes are rules, like the legal-move generator the search already uses, not engine knowledge (KataGo
# feeds Go's liberties and ladders the same way).  Without them the network has to derive every attack from piece
# placement before it can notice that a piece hangs - and hanging pieces, getting mated and failing to convert
# were what its games were lost to at Elo ~700.
# TRIED (A/B fork from iteration 126 of fly6, one hour per arm, --attack_planes): a tie - 41.2% with the planes against
# 42.1% without, over 600 games versus the same two ladder rungs (runs/ab2/head_to_head.txt).  They stay behind the
# flag, off: the features are always computed, but a network without the flag never reads them.
N_MOVES = 64 * 64 + 8 * 3 * 3   # from*64+to (queen promotion implied) + underpromotions (file, direction, piece)


def _flip(bb: int) -> int:
    return chess.flip_vertical(bb)


def encode(board: chess.Board) -> np.ndarray:
    us, them = board.turn, not board.turn
    flip = us == chess.BLACK
    bbs = [board.pieces_mask(pt, c) for c in (us, them) for pt in chess.PIECE_TYPES]
    bbs.append(chess.BB_SQUARES[board.ep_square] if board.ep_square is not None else 0)
    if flip:
        bbs = [_flip(b) for b in bbs]
    planes = np.unpackbits(np.array(bbs, dtype="<u8").view(np.uint8), bitorder="little")
    x = np.empty(N_FEATURES, dtype=np.float32)
    x[:13 * 64] = planes
    x[13 * 64:N_BASE] = (board.has_kingside_castling_rights(us), board.has_queenside_castling_rights(us),
                         board.has_kingside_castling_rights(them), board.has_queenside_castling_rights(them),
                         min(board.halfmove_clock, 100) / 100.0, board.is_repetition(2))
    x[N_BASE:] = _attack_features(board, us, flip)
    return x


def _attack_features(board: chess.Board, us: chess.Color, flip: bool) -> np.ndarray:
    att = []
    for colour in (us, not us):
        bb = 0
        for sq in chess.scan_forward(board.occupied_co[colour]):
            bb |= board.attacks_mask(sq)
        att.append(_flip(bb) if flip else bb)
    out = np.empty(129, dtype=np.float32)
    out[:128] = np.unpackbits(np.array(att, dtype="<u8").view(np.uint8), bitorder="little")
    out[128] = bool(att[1] & (_flip(board.pieces_mask(chess.KING, us)) if flip else board.pieces_mask(chess.KING, us)))
    return out


def upgrade_features(x: np.ndarray) -> np.ndarray:
    """A feature vector from before the attack planes -> the current layout.  The planes are in the side-to-move
    frame (we are 'white', moving up), and attacks depend on piece placement only, so they can be rebuilt exactly."""
    if len(x) == N_FEATURES:
        return x
    board = chess.Board(None)
    planes = np.asarray(x[:768], dtype=np.float32).reshape(12, 64)
    for p, sq in zip(*np.nonzero(planes)):
        board.set_piece_at(int(sq), chess.Piece(int(p) % 6 + 1, chess.WHITE if p < 6 else chess.BLACK))
    out = np.empty(N_FEATURES, dtype=x.dtype)
    out[:N_BASE] = x
    out[N_BASE:] = _attack_features(board, chess.WHITE, False)
    return out


def move_index(move: chess.Move, turn: chess.Color) -> int:
    f, t = move.from_square, move.to_square
    if turn == chess.BLACK:
        f, t = f ^ 56, t ^ 56
    if move.promotion and move.promotion != chess.QUEEN:
        ff = f & 7
        return 4096 + (ff * 3 + ((t & 7) - ff + 1)) * 3 + (move.promotion - chess.KNIGHT)
    return f * 64 + t


def legal_move_indices(board: chess.Board) -> tuple[list[chess.Move], np.ndarray]:
    moves = list(board.legal_moves)
    return moves, np.fromiter((move_index(m, board.turn) for m in moves), dtype=np.int64, count=len(moves))


def move_squares() -> tuple[np.ndarray, np.ndarray]:
    """From- and to-square of every policy index (in the side-to-move frame, like the indices themselves)."""
    frm, to = np.zeros(N_MOVES, dtype=np.int64), np.zeros(N_MOVES, dtype=np.int64)
    frm[:4096], to[:4096] = np.arange(4096) // 64, np.arange(4096) % 64
    for ff in range(8):
        for d in range(3):
            for piece in range(3):
                i = 4096 + (ff * 3 + d) * 3 + piece
                frm[i], to[i] = 48 + ff, 56 + min(max(ff + d - 1, 0), 7)       # a pawn on the 7th rank underpromoting
    return frm, to
