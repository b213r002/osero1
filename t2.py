import pandas as pd
import numpy as np

# csvの読み込み
csv_data = pd.read_csv("2.csv")

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
