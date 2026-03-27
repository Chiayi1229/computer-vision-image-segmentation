import os
import pandas as pd
from pycocotools.coco import COCO
import cv2
import numpy as np

# 初始化COCO
annotations_path = '/root/turtles-data/data/annotations.json'
coco = COCO(annotations_path)
prefix_dir = '/root/turtles-data/data'

# 读取CSV文件
csv_path = '/root/turtles-data/data/metadata_splits.csv'
df = pd.read_csv(csv_path)

# 确保CSV包含'id'和'file_name'列
if not {'id', 'file_name'}.issubset(df.columns):
    raise ValueError("CSV文件必须包含'id'和'file_name'列")

def get_binary_mask(coco, image_id):
    """
    生成二进制掩码图像，将所有类别合并为一个类。
    背景为0，前景为1。
    """
    # 获取所有类别的ID
    cat_ids = coco.getCatIds()
    # 获取与图像相关的所有标注ID
    ann_ids = coco.getAnnIds(imgIds=image_id, catIds=cat_ids, iscrowd=None)
    anns = coco.loadAnns(ann_ids)
    
    # 初始化掩码为全零（背景）
    height = coco.imgs[image_id]['height']
    width = coco.imgs[image_id]['width']
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # 遍历所有标注，将前景区域设为1
    for ann in anns:
        submask = coco.annToMask(ann)
        mask = np.maximum(mask, submask)
    
    return mask

def mask_to_rgb(mask):
    """
    将单通道掩码转换为RGB图像。
    背景（0） -> (0, 0, 0)
    前景（1） -> (255, 255, 255)
    """
    # 创建一个空的RGB图像
    mask_rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    # 设置前景为白色
    mask_rgb[mask == 1] = [255, 255, 255]
    return mask_rgb

def main():
    folder_path = prefix_dir
    annotations_dir = os.path.join(prefix_dir, 'annotations')
    os.makedirs(annotations_dir, exist_ok=True)
    
    total_images = len(df)
    print(f"Total images to process: {total_images}")
    
    for idx, row in df.iterrows():
        img_id = row['id']
        file_name = row['file_name']
        fname = os.path.join(prefix_dir, file_name)
        
        if not os.path.exists(fname):
            print(f"Image file {fname} does not exist. Skipping.")
            continue
        
        try:
            # 生成二进制掩码
            mask = get_binary_mask(coco, img_id)
            # 将掩码转换为RGB图像
            mask_rgb = mask_to_rgb(mask)
            # 保存路径
            mask_filename = os.path.splitext(os.path.basename(file_name))[0] + '.png'  # 确保保存为PNG
            mask_path = os.path.join(annotations_dir, mask_filename)
            # 使用OpenCV保存为PNG，确保保存为RGB
            cv2.imwrite(mask_path, cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR))
        except Exception as e:
            print(f"Error processing image {file_name} (ID: {img_id}): {e}")
            continue
        
        if (idx + 1) % 100 == 0 or (idx + 1) == total_images:
            print(f"Processed {idx + 1}/{total_images} images")
    
    print("All masks have been saved successfully.")

if __name__ == "__main__":
    main()
