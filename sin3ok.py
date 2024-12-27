import pandas as pd
import re
import numpy as np
import tensorflow as tf
import reversi
from tensorflow import keras
import os
from tensorflow.keras import layers

# CSVから試合内容を読み込むクラス
class MatchLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_match_info(self):
        csv_data = pd.read_csv(self.file_path, header=None)
        move_sequences = csv_data.iloc[:, -1]
        extract_one_hand = move_sequences.str.extractall(r'(..)')

        one_hand_df = extract_one_hand.reset_index().rename(
        columns={"level_0": "tournamentId", "level_1": "match", 0: "move_str"})

        conv_table = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6, "g": 7, "h": 8}
        one_hand_df["move"] = one_hand_df["move_str"].apply(lambda x: self.convert_move(x, conv_table))
            
        return one_hand_df
        # csv_data = pd.read_csv(self.file_path, header=None)

        # # 正規表現を使って2文字ずつ切り出す
        # extract_one_hand = csv_data[0].str.extractall('(..)')
        # one_hand_df = extract_one_hand.reset_index().rename(columns={"level_0": "tournamentId", "match": "move_no", 0: "move_str"})
        
        # # アルファベットを数字に変換するテーブル
        # conv_table = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6, "g": 7, "h": 8}
        # one_hand_df["move"] = one_hand_df.apply(lambda x: self.convert_move(x["move_str"], conv_table), axis=1)
        # return one_hand_df

    def convert_move(self, v, conv_table):
        l = conv_table[v[:1]]  # 列の値を変換
        r = int(v[1:])         # 行の値を変換
        return np.array([l - 1, r - 1], dtype='int8')

class ReversiProcessor:
    def __init__(self):
        self.now_tournament_id = -1
        self.turn_color = 1
        self.table_info = [0] * 100
        self.table_info[44], self.table_info[45] = 2, 1
        self.table_info[54], self.table_info[55] = 1, 2
        self.my_board_infos = []
        self.enemy_board_infos = []
        self.my_put_pos = []
        self.enemy_put_pos = []


    def process_tournament(self, df):
        # get_valid_moves メソッドを追加して、turn_color と table_info を使用
        def get_valid_moves(turn_color, table_info):
            directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
            valid_moves = []

            for y in range(1, 9):  # 1〜8 (10x10のリストのうち有効な範囲)
                for x in range(1, 9):
                    if table_info[x + y * 10] == 0:  # 空白マスか確認
                        for dx, dy in directions:
                            nx, ny = x + dx, y + dy
                            while 1 <= nx <= 8 and 1 <= ny <= 8 and table_info[nx + ny * 10] == 3 - turn_color:
                                nx += dx
                                ny += dy
                            if 1 <= nx <= 8 and 1 <= ny <= 8 and table_info[nx + ny * 10] == turn_color:
                                valid_moves.append((x, y))
                                break
            return valid_moves if valid_moves else None
        
        def PutStone(index, color, board):
            # indexは変数、colorは置く色、boardはボード情報を表すリスト
            if board[index] == 0:  # 空白マスの場合
                board[index] = color
                # フリップする方向（上下左右、斜め方向）
                directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

                for dx, dy in directions:
                    temp_index = index
                    temp_color = color
                    flip_positions = []

                    while True:
                        temp_index += dx + dy * 10
                        if not (0 <= temp_index < len(board)) or board[temp_index] == 0:
                            break
                        if board[temp_index] == temp_color:
                            break
                        flip_positions.append(temp_index)
                        temp_color = 3 - temp_color  # 次の色へ

                    # フリップするマスがある場合、実行
                    if flip_positions:
                        for pos in flip_positions:
                            board[pos] = color

        
        if df["tournamentId"] != self.now_tournament_id:
            self.reset_board(df["tournamentId"])
        else:
            self.turn_color = 1 if self.turn_color == 2 else 2

        valid_moves = get_valid_moves(self.turn_color, self.table_info)
        if valid_moves is None or len(valid_moves) == 0:
            self.turn_color = 1 if self.turn_color == 2 else 2

        put_pos = df["move"]
        self.record_training_data(put_pos)
        put_index = put_pos[0] + 1 + (put_pos[1] + 1) * 10
        PutStone(put_index, self.turn_color, self.table_info)

    def reset_board(self, tournament_id):
        self.now_tournament_id = tournament_id
        self.table_info = [0] * 100
        self.table_info[44], self.table_info[45] = 2, 1
        self.table_info[54], self.table_info[55] = 1, 2
        self.turn_color = 1

    def record_training_data(self, put_pos):
        my_board_info = np.zeros((8, 8), dtype="int8")
        enemy_board_info = np.zeros((8, 8), dtype="int8")
        for i in range(11, 89):
            if i % 10 == 0 or i % 10 == 9:
                continue
            if self.table_info[i] == 1:
                my_board_info[i // 10 - 1][i % 10 - 1] = 1
            elif self.table_info[i] == 2:
                enemy_board_info[i // 10 - 1][i % 10 - 1] = 1
        move_one_hot = np.zeros((8, 8), dtype='int8')
        move_one_hot[put_pos[1]][put_pos[0]] = 1
        if self.turn_color == 1:
            self.my_board_infos.append(np.array([my_board_info, enemy_board_info]))
            self.my_put_pos.append(move_one_hot)
        else:
            self.enemy_board_infos.append(np.array([enemy_board_info, my_board_info]))
            self.enemy_put_pos.append(move_one_hot)



# モデルクラス
class ReversiModel:
    def __init__(self, my_board_infos, enemy_board_infos, my_put_pos, enemy_put_pos):
        self.my_board_infos = my_board_infos
        self.enemy_board_infos = enemy_board_infos
        self.my_put_pos = my_put_pos
        self.enemy_put_pos = enemy_put_pos

    def create_model(self):
        model = keras.Sequential()
        model.add(layers.Permute((2, 3, 1), input_shape=(2, 8, 8)))
        for _ in range(12):
            model.add(layers.Conv2D(128, kernel_size=3, padding='same', activation='relu'))
        model.add(layers.Conv2D(1, kernel_size=1, use_bias=False))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='softmax'))
        model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def training(self):
        model = self.create_model()
        x_train = np.concatenate([self.my_board_infos, self.enemy_board_infos])
        y_train = np.concatenate([self.my_put_pos, self.enemy_put_pos]).reshape(-1, 64)

        try:
            tb_cb = keras.callbacks.TensorBoard(log_dir='model_log_sin', histogram_freq=1, write_graph=True)
            model.fit(x_train, y_train, epochs=800, batch_size=32, validation_split=0.2, callbacks=[tb_cb])
        except KeyboardInterrupt:
            os.makedirs('saved_model_reversi', exist_ok=True)
            model.save('saved_model_reversi/my_model_interrupt')
            print("学習が中断されました。モデルを保存しました。")
        model.save('saved_model_reversi/my_model')
        print("学習完了")

if __name__ == "__main__":
    match_loader = MatchLoader("2.csv")
    one_hand_df = match_loader.load_match_info()

    reversi_processor = ReversiProcessor()
    for _, row in one_hand_df.iterrows():
        reversi_processor.process_tournament(row)

    reversi_model = ReversiModel(
        reversi_processor.my_board_infos,
        reversi_processor.enemy_board_infos,
        reversi_processor.my_put_pos,
        reversi_processor.enemy_put_pos
    )
    reversi_model.training()
