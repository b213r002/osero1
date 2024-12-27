import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import random
import time

# MatchLoader: WTHORから試合データを読み込む
class MatchLoader:
    def __init__(self, csv_file):
        self.csv_file = csv_file
    
    def load_match_info(self):
        """CSVファイルから試合内容を読み込む"""
        try:
            csv_data = pd.read_csv(self.csv_file, header=None)
            move_sequences = csv_data.iloc[:, -1]
            extract_one_hand = move_sequences.str.extractall(r'(..)')

            one_hand_df = extract_one_hand.reset_index().rename(
                columns={"level_0": "tournamentId", "level_1": "match", 0: "move_str"}
            )

            conv_table = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6, "g": 7, "h": 8}
            one_hand_df["move"] = one_hand_df["move_str"].apply(lambda x: self.convert_move(x, conv_table))
            
            return one_hand_df
        except Exception as e:
            print(f"試合情報の読み込み中にエラーが発生しました: {e}")
            return None

    def convert_move(self, move_str, conv_table):
        """1手を数値に変換する"""
        col = conv_table[move_str[0]]
        row = int(move_str[1])
        return np.array([col - 1, row - 1], dtype='int8')

# ReversiProcessor: オセロの進行を処理
class ReversiProcessor:
    def __init__(self):
        self.table_info = np.zeros(100, dtype='int8')  # 10x10のボード
        self.my_board_infos = []
        self.enemy_board_infos = []
        self.my_put_pos = []
        self.enemy_put_pos = []
        self.turn_color = 1
        self.now_tournament_id = None

    def process_tournament(self, df):
        """試合の進行を処理"""
        if df["tournamentId"] != self.now_tournament_id:
            self.reset_board()
            self.now_tournament_id = df["tournamentId"]
        else:
            self.turn_color = 1 if self.turn_color == 2 else 2

        if not self.GetCanPutPos(self.turn_color):
            self.turn_color = 1 if self.turn_color == 2 else 2

        put_pos = df["move"]
        print(f"Processing move: {put_pos}, Turn: {self.turn_color}")
        self.record_training_data(put_pos)
        self.PutStone(put_pos)
        self.debug_print_board()
    
    def debug_print_board(self):
    #現在の盤面を表示
        print("Current Board:")
        for y in range(1, 9):
            row = [self.table_info[y * 10 + x] for x in range(1, 9)]
            print(' '.join(map(str, row)))
        print()

    def reset_board(self):
        """盤面をリセットする"""
        self.table_info.fill(0)
        self.table_info[44] = 2
        self.table_info[45] = 1
        self.table_info[54] = 1
        self.table_info[55] = 2
    
    def record_training_data(self, put_pos):
        """訓練用データを記録"""
        my_board_info = np.zeros((8, 8), dtype="int8")
        enemy_board_info = np.zeros((8, 8), dtype="int8")

        for i in range(11, 89):
            if i % 10 in {0, 9}:
                continue
            board_x = (i % 10) - 1
            board_y = (i // 10) - 1

            if self.table_info[i] == 1:
                my_board_info[board_y][board_x] = 1
            elif self.table_info[i] == 2:
                enemy_board_info[board_y][board_x] = 1

        move_one_hot = np.zeros((8, 8), dtype='int8')
        # put_posの値を確認し、範囲外の値でないことを確認
        assert 0 <= put_pos[0] <= 7 and 0 <= put_pos[1] <= 7, "put_posの値が範囲外です"
        move_one_hot[put_pos[1]][put_pos[0]] = 1

        if self.turn_color == 1:
            self.my_board_infos.append(np.array([my_board_info, enemy_board_info], dtype="int8"))
            self.my_put_pos.append(move_one_hot)
        else:
            self.enemy_board_infos.append(np.array([enemy_board_info, my_board_info], dtype="int8"))
            self.enemy_put_pos.append(move_one_hot)

        # # 1手ごとの盤面出力
        # print(f"Turn: {self.turn_color}")
        # print(f"黒●'s Board:\n{my_board_info}")
        # print(f"白○'s Board:\n{enemy_board_info}\n")

    def GetCanPutPos(self, turn_color):
        """置ける場所をリストとして返す"""
        return [pos for pos in range(100) if self.table_info[pos] == 0]

    # def PutStone(self, put_pos):
    #     """石を置く処理"""
    #     # put_index = put_pos[0] + put_pos[1] * 10
    #     # self.table_info[put_index] = self.turn_color
    #     # 8x8の座標を10x10のtable_info用に変換
    #     put_index = (put_pos[1] + 1) * 10 + (put_pos[0] + 1)
    #     self.table_info[put_index] = self.turn_color

    def PutStone(self, put_pos):
        #石を置き、挟んだ石を反転する処理
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        x, y = put_pos
        put_index = (y + 1) * 10 + (x + 1)  # 10x10のインデックスに変換
        self.table_info[put_index] = self.turn_color  # 石を置く

        # 8方向に対して反転チェック
        for dx, dy in directions:
            tmp_x, tmp_y = x, y
            reverse_list = []

            while True:
                tmp_x += dx
                tmp_y += dy
                tmp_index = (tmp_y + 1) * 10 + (tmp_x + 1)

                # ボード範囲外または空きマスの場合は終了
                if tmp_x < 0 or tmp_x > 7 or tmp_y < 0 or tmp_y > 7 or self.table_info[tmp_index] == 0:
                    break

                # 自分の石が見つかれば反転対象を確定
                if self.table_info[tmp_index] == self.turn_color:
                    for rev_index in reverse_list:
                        self.table_info[rev_index] = self.turn_color
                    break

                # 相手の石なら反転候補に追加
                reverse_list.append(tmp_index)


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

        tb_cb = keras.callbacks.TensorBoard(log_dir='model_log_sin', histogram_freq=1, write_graph=True)
        model.fit(x_train, y_train, epochs=800, batch_size=32, validation_split=0.2, callbacks=[tb_cb])
        model.save('saved_model_reversi/my_model')
        print("学習完了")


if __name__ == "__main__":
    match_loader = MatchLoader("2.csv")
    one_hand_df = match_loader.load_match_info()

    reversi_processor = ReversiProcessor()
    for _, row in one_hand_df.iterrows():
        reversi_processor.process_tournament(row)

    # Processorからモデルにトレーニングデータを渡す
    reversi_model = ReversiModel(
        reversi_processor.my_board_infos,
        reversi_processor.enemy_board_infos,
        reversi_processor.my_put_pos,
        reversi_processor.enemy_put_pos
    )

    # モデルを訓練
    reversi_model.training()
