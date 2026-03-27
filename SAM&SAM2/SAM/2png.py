import os
import cv2
from concurrent.futures import ProcessPoolExecutor, as_completed

# 定义源文件夹和目标文件夹
source_folder = '/root/turtles-data/data/images_jihe'
target_folder = '/root/autodl-tmp/train/images'

# 如果目标文件夹不存在，则创建
if not os.path.exists(target_folder):
    os.makedirs(target_folder)

# 支持的图片后缀
extensions = ('.jpg', '.jpeg', '.JPG')

def convert_image_cv(filename):
    source_path = os.path.join(source_folder, filename)
    try:
        # 读取图片
        img = cv2.imread(source_path)
        if img is None:
            raise ValueError("无法读取图片文件。")
        # 保存为PNG格式
        base_name = os.path.splitext(filename)[0]
        target_filename = base_name + '.png'
        target_path = os.path.join(target_folder, target_filename)
        cv2.imwrite(target_path, img)
        return f"已转换: {filename} -> {target_filename}"
    except Exception as e:
        return f"转换失败: {filename}. 错误: {e}"

def main():
    # 获取所有符合条件的文件
    files = [f for f in os.listdir(source_folder) if f.endswith(extensions)]
    total_files = len(files)
    print(f"共找到 {total_files} 张图片需要转换。")

    # 设置进程池，建议根据CPU核心数调整
    max_workers = os.cpu_count() or 4
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_file = {executor.submit(convert_image_cv, file): file for file in files}
        for future in as_completed(future_to_file):
            result = future.result()
            print(result)

    print("所有图片转换完成。")

if __name__ == "__main__":
    main()
