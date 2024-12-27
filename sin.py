import pandas as pd
import re
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import reversi
import time
import matplotlib.pyplot as plt
import reversi

print(dir(reversi))

class ReversiTrainer:
    def __init__(self):
        # 属性の初期化
        self.my_board_infos = []
        self.enemy_board_infos = []
        self.my_put_pos = []
        self.enemy_put_pos = []
        self.table_info = [0] * 100  # 初期盤面
        self.turn_color = 1  # 黒番で開始
        self.now_tournament_id = None  # 現在の試合IDを管理

    # CSVから試合内容を読み込む
    def load_match_info(self):
        csv_data = pd.read_csv("2.csv")
        move_sequences = csv_data.iloc[:, -1]
        extract_one_hand = move_sequences.str.extractall(r'(..)')  
        one_hand_df = extract_one_hand.reset_index().rename(columns={"level_0": "tournamentId", "level_1": "match", 0: "move_str"})

        conv_table = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6, "g": 7, "h": 8}
        one_hand_df["move"] = one_hand_df["move_str"].apply(lambda x: self.convert_move(x, conv_table))

        print(csv_data.head())  # データの先頭を表示
        print(f"Total moves: {len(one_hand_df)}")

        return one_hand_df

    # 1手を数値に変換する
    def convert_move(self, v, conv_table):
        l = conv_table[v[:1]]  # 列の値を変換する
        r = int(v[1:])         # 行の値を変換する
        return np.array([l - 1, r - 1], dtype='int8')

    def get_valid_moves(self, turn_color, table_info):
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

        return valid_moves

    def process_tournament(self, df):
        print(f"my_board_infos: {len(self.my_board_infos)}, enemy_board_infos: {len(self.enemy_board_infos)}")

        if df["tournamentId"] != self.now_tournament_id:
            self.table_info = [0] * 100
            self.table_info[44] = 2
            self.table_info[45] = 1
            self.table_info[54] = 1
            self.table_info[55] = 2
            self.turn_color = 1
            self.now_tournament_id = df["tournamentId"]
        else:
            self.turn_color = 1 if self.turn_color == 2 else 2

        if len(self.get_valid_moves(self.turn_color, self.table_info)) == 0:
            self.turn_color = 1 if self.turn_color == 2 else 2

        put_pos = df["move"]
        self.record_training_data(put_pos)

        put_index = put_pos[0] + 1 + (put_pos[1] + 1) * 10
        reversi.PutStone(put_index, self.turn_color, self.table_info)

    def record_training_data(self, put_pos):
        my_board_info = np.zeros(shape=(8,8), dtype="int8")
        enemy_board_info = np.zeros(shape=(8,8), dtype="int8")
        for i in range(len(self.table_info)):
            if i >= 0 and i <= 9:
                continue
            if i / 10 == 0:
                continue
            if i / 9 == 0:
                continue
            if i >= 90 and i <= 99:
                continue

            if self.table_info[i] == 1:
                my_board_info[int(i/10) - 1][int(i%10) - 1] = 1
            elif self.table_info[i] == 2:
                enemy_board_info[int(i/10) - 1][int(i%10) - 1] = 1

        move_one_hot = np.zeros(shape=(8,8), dtype='int8')
        move_one_hot[put_pos[1]][put_pos[0]] = 1

        if self.turn_color == 1:
            self.my_board_infos.append(np.array([my_board_info.copy(), enemy_board_info.copy()], dtype="int8"))
            self.my_put_pos.append(move_one_hot)
        else:
            self.enemy_board_infos.append(np.array([enemy_board_info.copy(), my_board_info.copy()], dtype="int8"))
            self.enemy_put_pos.append(move_one_hot)

    def create_model(self):
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
        model.add(layers.Conv2D(128, kernel_size=3,padding='same',activation='relu'))
        model.add(layers.Conv2D(1, kernel_size=1,use_bias=False))
        model.add(layers.Flatten())
        model.add(Bias((1, 64)))
        model.add(layers.Activation('softmax'))

        model.compile(keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False), 'categorical_crossentropy', metrics=['accuracy'])
        return model

    def training(self):
        model = self.create_model()  # モデルの生成
        print("Training started...")

        match_data = self.load_match_info()
        print(f"Total matches: {match_data['tournamentId'].nunique()}")
        match_data.apply(self.process_tournament, axis=1)

        print(f"my_board_infos size: {len(self.my_board_infos)}")
        print(f"enemy_board_infos size: {len(self.enemy_board_infos)}")
        print(f"my_put_pos size: {len(self.my_put_pos)}")
        print(f"enemy_put_pos size: {len(self.enemy_put_pos)}")
        x_train = np.concatenate([self.my_board_infos, self.enemy_board_infos])
        y_train_tmp = np.concatenate([self.my_put_pos, self.enemy_put_pos])

        y_train = y_train_tmp.reshape(-1, 64)

        print(f"x_train shape: {x_train.shape}")
        print(f"y_train shape: {y_train.shape}")

        try:
            tb_cb = keras.callbacks.TensorBoard(log_dir='model_log/relu_12', histogram_freq=1, write_graph=True)
            model.fit(x_train, y_train, epochs=1000, batch_size=32, validation_split=0.2, callbacks=[tb_cb])
        except KeyboardInterrupt:
            model.save('saved_model_reversi/my_model_interrupt')
            print('Output saved')
            return

        model.save('saved_model_reversi/my_model')
        print('complete')

if __name__ == "__main__":
    trainer = ReversiTrainer()
    trainer.training()
