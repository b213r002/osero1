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

# 試験的な実装例として、簡単な動作テストを行う
if __name__ == "__main__":
    board = [0] * 100
    board[44], board[45] = 2, 1
    board[54], board[55] = 1, 2
    PutStone(45, 1, board)
    print(board)  # 期待される出力：[0, 0, ..., 0, 1, 2, 1, 2, 1, ...] (置かれた石が反映された状態)
