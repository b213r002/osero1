import os

# モデルファイルのパス
original_model_path = "saved_model_reversi/my_modelryoua3"
quantized_model_path = "saved_model_reversi/my_modelryoua3_quantized.tflite"

# ファイルサイズを取得
original_size = sum(os.path.getsize(os.path.join(dp, f)) for dp, dn, filenames in os.walk(original_model_path) for f in filenames)
quantized_size = os.path.getsize(quantized_model_path)

# サイズを表示 (バイト -> キロバイト)
print(f"Original model size: {original_size / 1024:.2f} KB")
print(f"Quantized model size: {quantized_size / 1024:.2f} KB")
