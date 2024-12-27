import numpy as np
import tensorflow as tf
from tensorflow import keras
import random

model = keras.models.load_model('saved_model/my_model5000@1')

# 盤面の初期化
def init_board():
    board = np.zeros((8, 8), dtype=int)
    board[3, 3] = board[4, 4] = -1  # 白
    board[3, 4] = board[4, 3] = 1   # 黒
    return board

# 石を置けるか確認する関数
def can_put(board, row, col, is_black_turn):
    if board[row, col] != 0:
        return False
    player = 1 if is_black_turn else -1
    opponent = -player
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    for dx, dy in directions:
        x, y = row + dx, col + dy
        found_opponent = False
        while 0 <= x < 8 and 0 <= y < 8 and board[x, y] == opponent:
            found_opponent = True
            x += dx
            y += dy
        if found_opponent and 0 <= x < 8 and 0 <= y < 8 and board[x, y] == player:
            return True
    return False

# 石を置く関数
def put(board, row, col, is_black_turn):
    player = 1 if is_black_turn else -1
    board[row, col] = player
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    for dx, dy in directions:
        x, y = row + dx, col + dy
        stones_to_flip = []
        while 0 <= x < 8 and 0 <= y < 8 and board[x, y] == -player:
            stones_to_flip.append((x, y))
            x += dx
            y += dy
        if 0 <= x < 8 and 0 <= y < 8 and board[x, y] == player:
            for fx, fy in stones_to_flip:
                board[fx, fy] = player

# モデルAIによる手を選ぶ関数
def model_move(board, model, is_black_turn):
    valid_moves = [(row, col) for row in range(8) for col in range(8) if can_put(board, row, col, is_black_turn)]
    if valid_moves:
        board_input = np.zeros((1, 2, 8, 8), dtype='int8')
        board_input[0, 0] = (board == 1).astype('int8')  # 黒の盤面
        board_input[0, 1] = (board == -1).astype('int8')  # 白の盤面
        predictions = model.predict(board_input)[0]
        
        best_move = max(valid_moves, key=lambda pos: predictions[pos[0] * 8 + pos[1]])
        return best_move
    return None
                
# # モデルAI(ランダム性を含む)による手を選ぶ関数
# def model_move(board, model, is_black_turn):
#     valid_moves = [(row, col) for row in range(8) for col in range(8) if can_put(board, row, col, is_black_turn)]
#     if valid_moves:
#         board_input = np.zeros((1, 2, 8, 8), dtype='int8')
#         board_input[0, 0] = (board == 1).astype('int8')  # 黒の盤面
#         board_input[0, 1] = (board == -1).astype('int8')  # 白の盤面
#         predictions = model.predict(board_input)[0]

#         # 有効手の確率を抽出
#         valid_probabilities = np.array([predictions[row * 8 + col] for row, col in valid_moves])
#         valid_probabilities /= valid_probabilities.sum()  # 確率を正規化
#         # 確率に基づいてランダムに手を選択
#         chosen_index = np.random.choice(len(valid_moves), p=valid_probabilities)
#         return valid_moves[chosen_index]
#     return None


# Minimax AIの盤面評価関数
def evaluate_board(board, is_black_turn):
    black_score = np.sum(board == 1)
    white_score = np.sum(board == -1)
    return black_score - white_score if is_black_turn else white_score - black_score

# Minimaxアルゴリズムによる手を選ぶ関数
def minimax(board, depth, is_black_turn, alpha, beta):
    if depth == 0 or np.all(board != 0):
        return evaluate_board(board, is_black_turn), None

    best_move = None
    if is_black_turn:
        max_eval = float('-inf')
        for row in range(8):
            for col in range(8):
                if can_put(board, row, col, True):
                    new_board = board.copy()
                    put(new_board, row, col, True)
                    eval, _ = minimax(new_board, depth - 1, False, alpha, beta)
                    if eval > max_eval:
                        max_eval = eval
                        best_move = (row, col)
                    alpha = max(alpha, eval)
                    if beta <= alpha:
                        break
        return max_eval, best_move
    else:
        min_eval = float('inf')
        for row in range(8):
            for col in range(8):
                if can_put(board, row, col, False):
                    new_board = board.copy()
                    put(new_board, row, col, False)
                    eval, _ = minimax(new_board, depth - 1, True, alpha, beta)
                    if eval < min_eval:
                        min_eval = eval
                        best_move = (row, col)
                    beta = min(beta, eval)
                    if beta <= alpha:
                        break
        return min_eval, best_move

# 勝者判定のために石の数をカウントする関数
def count_stones(board):
    black = np.sum(board == 1)
    white = np.sum(board == -1)
    return black, white

# 盤面を表示する関数
def print_board(board):
    for row in board:
        print(" ".join(["●" if x == 1 else "○" if x == -1 else "." for x in row]))
    print()


# 対戦の進行を管理する関数（modelAI vs Minimax AI）
def play_game_model_vs_minimax():
    board = init_board()
    is_black_turn = True  # 黒がモデルAI、白がMinimax AIとします

    while True:
        print_board(board)
        if is_black_turn:
            move = model_move(board, model, is_black_turn)
        else:
            _, move = minimax(board, 5, is_black_turn, float('-inf'), float('inf'))

        if move:
            put(board, move[0], move[1], is_black_turn)
        else:
            if not any(can_put(board, row, col, not is_black_turn) for row in range(8) for col in range(8)):
                break  # 両者ともに合法手がない場合は終了
            # パスの場合は何もしない
        is_black_turn = not is_black_turn  # ターンを切り替え

    return board


# 勝敗を記録する関数
def simulate_games(num_games):
    random_wins = 0
    minimax_wins = 0
    draws = 0

    for _ in range(num_games):
        final_board = play_game_model_vs_minimax()
        black, white = count_stones(final_board)
        if black > white:
            random_wins += 1
        elif white > black:
            minimax_wins += 1
        else:
            draws += 1

    print(f"モデルAI(黒)の勝利数: {random_wins}")
    print(f"Minimax AI(白)の勝利数: {minimax_wins}")
    print(f"引き分け数: {draws}")

# 対戦をシミュレーション
simulate_games(10)  # 100試合をシミュレーション