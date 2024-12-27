import pandas as pd
import re
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import reversi
import time
import matplotlib.pyplot as plt

# # CSVファイルの読み込み (ヘッダーなし)
# csv_data = pd.read_csv("10.csv", header=None)
# # 最後の列（ムーブシーケンス）だけを抽出
# move_sequences = csv_data.iloc[:, -1]  # 最後の列を抽出
# # 中間結果を新しいCSVファイルに保存
# move_sequences.to_csv('W2018.csv', index=False)
# # 中間結果を新しいCSVファイルに保存
# csv_data.to_csv('1.csv', index=False)

# CSVファイルの読み込み (ヘッダーなし)
csv_data = pd.read_csv("10n.csv", header=None)

# 正規表現を使って2文字ずつ切り出す
extract_one_hand = csv_data[0].str.extractall('(..)')
# 正規表現を使って2文字ずつ切り出す
# 正規表現を使って2文字ずつ切り出す
extract_one_hand.to_csv('1.csv', index=False)