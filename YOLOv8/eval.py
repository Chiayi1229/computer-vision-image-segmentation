
from ultralytics import YOLO
if __name__ == '__main__':
    # 加载模型
    model = YOLO("runs/segment/train/weights/best.pt")  # 加载预训练模型（建议用于训练）
    # model = YOLO("best.pt")
    metrics = model.val(data="D:\\Desktop\\ultralytics-8.2.103-turtle\\ultralytics-8.2.103-turtle\\data\\turtle.yaml")  # 在验证集上评估模型性能
    a = 1
