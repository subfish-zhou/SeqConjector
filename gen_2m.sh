
# 生成 200万 数据脚本
# 策略：
# 1. 基础阶段 (Easy): 50万条。难度 0.2。主要是短程序，让模型学会基础算术和简单逻辑。
# 2. 进阶阶段 (Medium): 100万条。难度 0.5。混合长度，引入更复杂的算子。
# 3. 高级阶段 (Hard): 50万条。难度 0.9。包含长程序和复杂嵌套。

# 确保目录存在
mkdir -p data_gen

echo "Generating Easy batch (500k)..."
python generate_data.py --out_dir data_gen --total_samples 500000 --difficulty 0.2 --prefix "easy" --seed 1000 --workers 12

echo "Generating Medium batch (1M)..."
python generate_data.py --out_dir data_gen --total_samples 1000000 --difficulty 0.5 --prefix "medium" --seed 2000 --workers 12

echo "Generating Hard batch (500k)..."
python generate_data.py --out_dir data_gen --total_samples 500000 --difficulty 0.9 --prefix "hard" --seed 3000 --workers 12

echo "Done. Total 2M samples ready in data_gen/"

