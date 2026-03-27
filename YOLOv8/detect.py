from ultralytics import YOLO

# 加载模型
model = YOLO("runs/detect/train/weights/best.pt")  # 加载预训练模型（推荐用于训练）

# Use the model
results = model.predict(source="images", save=True, save_crop=True)  # 训练模型