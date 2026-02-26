from ultralytics import YOLO
# 加载自己训练结束后的模型权重
model = YOLO(r'E:\Yolov11\YOLOv11\runs\detect\train\weights\last.pt')
# 推理的图像文件路径
source = r'E:\Yolov11\YOLOv11\data\Dataaugmentation=8\test\images'
# 设置参数
model.predict(source, save=True, imgsz=256, conf=0.5, line_width=2)