import os
import pandas as pd
import shutil

# 定义路径
prefix_dir = '/root/turtles-data/data'
csv_path = os.path.join(prefix_dir, 'metadata_splits.csv')
images_dir = os.path.join(prefix_dir, 'images_jihe')

# 创建images文件夹
os.makedirs(images_dir, exist_ok=True)

# 读取CSV文件
df = pd.read_csv(csv_path)

# 确保CSV包含'file_name'列
if 'file_name' not in df.columns:
    raise ValueError("CSV文件必须包含'file_name'列")

# 获取所有file_name
file_names = df['file_name'].tolist()
total_files = len(file_names)
print(f"Total images to copy: {total_files}")

# 遍历并复制图片
for idx, file_name in enumerate(file_names, 1):
    src_path = os.path.join(prefix_dir, file_name)
    dest_path = os.path.join(images_dir, os.path.basename(file_name))
    
    if not os.path.exists(src_path):
        print(f"Source file {src_path} does not exist. Skipping.")
        continue
    
    try:
        shutil.copy(src_path, dest_path)
    except Exception as e:
        print(f"Error copying {src_path} to {dest_path}: {e}")
        continue
    
    if idx % 100 == 0 or idx == total_files:
        print(f"Copied {idx}/{total_files} images")

print("All images have been copied successfully.")
