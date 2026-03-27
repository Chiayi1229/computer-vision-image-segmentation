import torch
import monai
from tqdm import tqdm
from statistics import mean
from torch.utils.data import DataLoader
from torch.optim import Adam
import src.utils as utils
from src.dataloader import DatasetSegmentation, collate_fn
from src.processor import Samprocessor
from src.segment_anything import build_sam_vit_b
from src.lora import LoRA_sam
import yaml
import numpy as np

def print_tensor_info(tensor, name):
    """Helper function to print tensor information"""
    print(f"\n{name} info:")
    print(f"Shape: {tensor.shape}")
    print(f"Type: {tensor.dtype}")
    print(f"Unique values: {torch.unique(tensor)}")
    print(f"Range: [{tensor.min()}, {tensor.max()}]")

def calculate_iou(pred, target, smooth=1e-6):
    """Calculate IoU with correct tensor types"""
    pred = pred.bool()
    target = target.bool()
    
    intersection = (pred & target).sum().float()
    union = (pred | target).sum().float()
    
    iou = (intersection + smooth) / (union + smooth)
    return iou.item()

# 加载配置文件
with open("/root/autodl-tmp/SAM-fine-tune/config.yaml", "r") as ymlfile:
    config_file = yaml.load(ymlfile, Loader=yaml.Loader)

# 类别映射
category_names = {1: "turtle", 2: "flipper", 3: "head"}

# 初始化模型和数据
train_dataset_path = config_file["DATASET"]["TRAIN_PATH"]
sam = build_sam_vit_b(checkpoint=config_file["SAM"]["CHECKPOINT"])
sam_lora = LoRA_sam(sam, config_file["SAM"]["RANK"])  
model = sam_lora.sam
processor = Samprocessor(model)
train_ds = DatasetSegmentation(config_file, processor, mode="train")

# 数据加载器
train_dataloader = DataLoader(
    train_ds, 
    batch_size=config_file["TRAIN"]["BATCH_SIZE"], 
    shuffle=True, 
    collate_fn=collate_fn,
    num_workers=0,
    pin_memory=False
)

# 优化器和损失函数
optimizer = Adam(model.image_encoder.parameters(), lr=config_file["TRAIN"]["LEARNING_RATE"], weight_decay=0)
seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
num_epochs = config_file["TRAIN"]["NUM_EPOCHS"]

# 设备设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
model.train().to(device)

# 训练循环
total_loss = []
best_miou = 0.0

try:
    for epoch in range(num_epochs):
        epoch_losses = []
        part_iou_scores = {1: [], 2: [], 3: []}
        
        for i, batch in enumerate(tqdm(train_dataloader)):
            try:
                # 前向传播
                outputs = model(batched_input=batch, multimask_output=False)
                stk_gt, stk_out = utils.stacking_batch(batch, outputs)
                
                if stk_gt is None or stk_out is None:
                    continue

                # 移动到设备并调整维度
                stk_gt = stk_gt.to(device)                    # 移动到GPU
                stk_out = stk_out.to(device)                  # 移动到GPU
                
                # 调整维度
                stk_out = stk_out.squeeze(2)                  # 移除多余的维度
                stk_gt = stk_gt.unsqueeze(1)                  # 添加通道维度

                # 打印第一个batch的调试信息
                if i == 0:
                    print_tensor_info(stk_gt, "Ground Truth (after reshape)")
                    print_tensor_info(stk_out, "Model Output (after reshape)")
                    print(f"GT shape: {stk_gt.shape}, Output shape: {stk_out.shape}")

                # 计算损失
                loss = seg_loss(stk_out, stk_gt.float())
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # 计算IoU
                pred_masks = (torch.sigmoid(stk_out) > 0.5).squeeze(1)  # 移除通道维度用于IoU计算
                stk_gt = stk_gt.squeeze(1)  # 移除通道维度用于IoU计算
                
                for part_id in part_iou_scores.keys():
                    part_gt = (stk_gt == part_id)
                    part_pred = pred_masks
                    
                    if i < 2:  # 打印前两个batch的信息
                        print(f"\nBatch {i}, Part {part_id}:")
                        print(f"GT sum: {part_gt.sum().item()}")
                        print(f"Pred sum: {part_pred.sum().item()}")
                    
                    try:
                        iou = calculate_iou(part_pred, part_gt)
                        part_iou_scores[part_id].append(iou)
                        
                        if i < 2:
                            print(f"IoU: {iou}")
                    except Exception as e:
                        print(f"Error calculating IoU: {str(e)}")
                        continue

                epoch_losses.append(loss.item())
                
                # 监控GPU内存
                if i % 100 == 0 and torch.cuda.is_available():
                    print(f"\nGPU Memory: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
                
            except Exception as e:
                print(f"\nError in batch {i}: {str(e)}")
                continue
        
        # 计算epoch结果
        if epoch_losses:
            mean_loss = mean(epoch_losses)
            print(f'\nEPOCH: {epoch}')
            print(f'Mean loss training: {mean_loss}')
        
            # 计算每个部位的IoU
            all_ious = []
            print("\nPer-part IoU scores:")
            for part_id, iou_list in part_iou_scores.items():
                if iou_list:
                    mean_iou = mean([x for x in iou_list if not np.isnan(x)])
                    all_ious.append(mean_iou)
                    print(f'{category_names[part_id]}: {mean_iou:.4f}')
                else:
                    print(f'{category_names[part_id]}: No valid IoU scores')

            # 计算并保存最佳模型
            if all_ious:
                miou = mean(all_ious)
                print(f'\nMean IoU (mIoU): {miou:.4f}')
                
                if miou > best_miou:
                    best_miou = miou
                    rank = config_file["SAM"]["RANK"]
                    save_path = f"best_model_rank{rank}_miou{miou:.4f}.safetensors"
                    sam_lora.save_lora_parameters(save_path)
                    print(f"Saved new best model with mIoU: {miou:.4f}")
            else:
                print("No valid IoU scores for this epoch")

        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # 保存最终模型
    rank = config_file["SAM"]["RANK"]
    sam_lora.save_lora_parameters(f"final_model_rank{rank}.safetensors")
    print(f"\nTraining completed! Best mIoU: {best_miou:.4f}")
    
except KeyboardInterrupt:
    print("\nTraining interrupted by user!")
    rank = config_file["SAM"]["RANK"]
    sam_lora.save_lora_parameters(f"interrupted_model_rank{rank}.safetensors")
    
except Exception as e:
    print(f"Training error: {str(e)}")
    
finally:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"Final GPU memory: {torch.cuda.memory_allocated()/1024**2:.1f}MB")