import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# csvの読み込み
def process_csv(file_path):
    csv_data = pd.read_csv(file_path)

    # 最終列から手のデータを抽出
    move_sequences = csv_data.iloc[:, -1]
    extract_one_hand = move_sequences.str.extractall(r'(..)')

    # Indexを再構成して、1行1手の表にする
    one_hand_df = extract_one_hand.reset_index().rename(columns={"level_0": "tournamentId", "level_1": "match", 0: "move_str"})

    # 列の値を数字に変換するdictonaryを作る
    def left_build_conv_table():
        left_table = ["a","b","c","d","e","f","g","h"]
        left_conv_table = {}
        n = 1
        for t in left_table:
            left_conv_table[t] = n
            n = n + 1
        return left_conv_table

    left_conv_table = left_build_conv_table()

    # dictionaryを使って列の値を数字に変換する
    def left_convert_colmn_str(col_str):
        return left_conv_table[col_str]  

    # 1手を数値に変換する
    def convert_move(v):
        l = left_convert_colmn_str(v[:1]) # 列の値を変換する
        r = int(v[1:]) # 行の値を変換する
        return np.array([l - 1, r - 1], dtype='int8')

    one_hand_df["move"] = one_hand_df["move_str"].apply(lambda x: convert_move(x))

    print(csv_data.head())  # データの先頭を表示
    print(f"Total moves: {len(one_hand_df)}")

    return one_hand_df

# 使用例
one_hand_df = process_csv("2.csv")

# 盤面の中にあるかどうかを確認する
def is_in_board(cur):
    return cur >= 0 and cur <= 7

# ある方向(direction）に対して石を置き、可能なら敵の石を反転させる
def put_for_one_move(board_a, board_b, move_row, move_col, direction):
    board_a[move_row][move_col] = 1

    tmp_a = board_a.copy()
    tmp_b = board_b.copy()
    cur_row = move_row
    cur_col = move_col

    cur_row += direction[0]
    cur_col += direction[1]
    reverse_cnt = 0
    while is_in_board(cur_row) and is_in_board(cur_col):
        if tmp_b[cur_row][cur_col] == 1: # 反転させる
            tmp_a[cur_row][cur_col] = 1
            tmp_b[cur_row][cur_col] = 0
            cur_row += direction[0]
            cur_col += direction[1]
            reverse_cnt += 1
        elif tmp_a[cur_row][cur_col] == 1:
            return tmp_a, tmp_b, reverse_cnt
        else:
            return board_a, board_b, reverse_cnt
    return board_a, board_b, reverse_cnt

# 方向の定義（配列の要素は←、↖、↑、➚、→、➘、↓、↙に対応している）
directions = [[-1,0],[-1,1],[0,1],[1,1],[1,0],[1,-1],[0,-1],[-1,-1]]

# ある位置に石を置く。すべての方向に対して可能なら敵の石を反転させる
def put(board_a, board_b ,move_row, move_col):
    tmp_a = board_a.copy()
    tmp_b = board_b.copy()
    global directions
    reverse_cnt_amount = 0
    for d in directions:
        board_a ,board_b, reverse_cnt = put_for_one_move(board_a, board_b, move_row, move_col, d)
        reverse_cnt_amount += reverse_cnt

    return board_a , board_b, reverse_cnt_amount

# 盤面の位置に石がないことを確認する
def is_none_state(board_a, board_b, cur_row, cur_col):
    return board_a[cur_row][cur_col] == 0 and board_b[cur_row][cur_col] == 0

# 盤面に石が置けるかを確認する（ルールでは敵の石を反転できるような位置にしか石を置けない）  
def can_put(board_a, board_b, cur_row, cur_col):
    copy_board_a = board_a.copy()
    copy_board_b = board_b.copy()
    _,  _, reverse_cnt_amount = put(copy_board_a, copy_board_b, cur_row, cur_col)
    return reverse_cnt_amount > 0

# パスする必要のある盤面かを確認する
def is_pass(is_black_turn, board_black, board_white):
    if is_black_turn:
        own = board_black
        opponent = board_white
    else:
        own = board_white
        opponent = board_black
    for cur_row in range(8):
        for cur_col in range(8):
            if is_none_state(own, opponent, cur_row, cur_col) and can_put(own, opponent, cur_row, cur_col):
                return False
    return True

# 変数の初期化
b_tournamentId = -1 # トーナメント番号
board_black = [] # 黒にとっての盤面の状態（１試合保存用）
board_white = [] # 白にとっての盤面の状態（１試合保存用）
boards_black = [] # 黒にとっての盤面の状態（全トーナメント保存用）
boards_white = [] # 白にとっての盤面の状態（全トーナメント保存用）
moves_black = [] # 黒の打ち手（全トーナメント保存用）
moves_white = [] # 白の打ち手（全トーナメント保存用）
is_black_turn = True # True = 黒の番、 False = 白の番
# ターン（黒の番 or 白の番）を切り変える
def switch_turn(is_black_turn):
    return is_black_turn == False # ターンを切り替え

# 棋譜のデータを１つ読み、学習用データを作成する関数
def process_tournament(df):
    global is_black_turn
    global b_tournamentId
    global boards_white
    global boards_black
    global board_white
    global board_black
    global moves_white
    global moves_black
    if df["tournamentId"] != b_tournamentId:
        # トーナメントが切り替わったら盤面を初期状態にする
        b_tournamentId = df["tournamentId"]
        board_black = np.zeros(shape=(8,8), dtype='int8')
        board_black[3][4] = 1
        board_black[4][3] = 1
        board_white = np.zeros(shape=(8,8), dtype='int8')
        board_white[3][3] = 1
        board_white[4][4] = 1
        is_black_turn = True
    else:
        # ターンを切り替える
        is_black_turn = switch_turn(is_black_turn)
        if is_pass(is_black_turn, board_black, board_white): # パスすべき状態か確認する
            is_black_turn = switch_turn(is_black_turn) #パスすべき状態の場合はターンを切り替える

    # 黒の番なら黒の盤面の状態を保存する、白の番なら白の盤面の状態を保存する
    if is_black_turn:
        boards_black.append(np.array([board_black.copy(), board_white.copy()], dtype='int8'))
    else:
        boards_white.append(np.array([board_white.copy(), board_black.copy()], dtype='int8'))
    
    # 打ち手を取得する
    move = df["move"]
    move_one_hot = np.zeros(shape=(8,8), dtype='int8')
    move_one_hot[move[1]][move[0]] = 1

    # 黒の番なら自分→敵の配列の並びを黒→白にして打ち手をセットする。白の番なら白→黒の順にして打ち手をセットする
    if is_black_turn:
        moves_black.append(move_one_hot)
        board_black, board_white, _ = put(board_black, board_white, move[1], move[0])
    else:
        moves_white.append(move_one_hot)
        board_white, board_black, _ = put(board_white, board_black, move[1], move[0])

# 棋譜データを学習データに展開する
one_hand_df.apply(lambda x: process_tournament(x), axis= 1)

x_train = np.concatenate([boards_black, boards_white])
y_train = np.concatenate([moves_black, moves_white])  
# 教師データは8x8の2次元データになっているので、64要素の1次元データにreshapeする
y_train_reshape = y_train.reshape(-1,64)

class Bias(keras.layers.Layer):
    def __init__(self, input_shape):
        super(Bias, self).__init__()
        self.W = tf.Variable(initial_value=tf.zeros(input_shape[1:]), trainable=True)

    def call(self, inputs):
        return inputs + self.W  

model = keras.Sequential()
model.add(layers.Permute((2,3,1), input_shape=(2,8,8)))
model.add(layers.Conv2D(128, kernel_size=3,padding='same',activation='relu'))
model.add(layers.Conv2D(128, kernel_size=3,padding='same',activation='relu'))
model.add(layers.Conv2D(128, kernel_size=3,padding='same',activation='relu'))
model.add(layers.Conv2D(128, kernel_size=3,padding='same',activation='relu'))
model.add(layers.Conv2D(128, kernel_size=3,padding='same',activation='relu'))
model.add(layers.Conv2D(128, kernel_size=3,padding='same',activation='relu'))
model.add(layers.Conv2D(128, kernel_size=3,padding='same',activation='relu'))
model.add(layers.Conv2D(128, kernel_size=3,padding='same',activation='relu'))
model.add(layers.Conv2D(128, kernel_size=3,padding='same',activation='relu'))
model.add(layers.Conv2D(128, kernel_size=3,padding='same',activation='relu'))
model.add(layers.Conv2D(128, kernel_size=3,padding='same',activation='relu'))
model.add(layers.Conv2D(1, kernel_size=1,use_bias=False))
model.add(layers.Flatten())
model.add(Bias((1, 64)))
model.add(layers.Activation('softmax'))
model.compile(
    optimizer=keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


# Tensor Boardコールバック
tb_cb = keras.callbacks.TensorBoard(log_dir='model_log/relu_12', histogram_freq=1, write_graph=True)
model.fit(x_train, y_train_reshape, epochs=1000, batch_size=32, validation_split=0.2,callbacks=[tb_cb])
model.save('saved_model_reversi/my_model')
print('Output saved')
