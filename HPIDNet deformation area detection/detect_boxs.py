import os
import json
import torch
from ultralytics import YOLO

# 加载 YOLOv8-seg 分割模型，并使用 GPU 进行推理
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO(r'E:\Yolov11\YOLOv11\runs\detect\Data augmentation=6\train=200\weights\best.pt')  # 请替换为你的模型路径

# 设置输入 BMP 图片目录和输出 TXT 目录
input_dir = r'E:\Yolov11\YOLOv11\data\Data augmentation=6\test\images'  # BMP 检测图像路径
output_dir = r'E:\Yolov11\YOLOv11\runs\detect\txt'  # 保存 txt 的目录

# 创建输出目录（如果不存在）
os.makedirs(output_dir, exist_ok=True)

# 定义推理函数
def infer_and_save_labels(image_path, model, output_dir):
    # 获取图片名称（不含扩展名）
    img_name = os.path.splitext(os.path.basename(image_path))[0]

    # 使用 YOLOv8 的 predict 进行推理
    results = model.predict(image_path, device=device, save=True)  # save=True 自动保存检测结果

    # 初始化结果列表
    detections = []

    # 遍历每个检测到的目标
    for r in results[0].boxes:  # 只取第一张图片的检测结果
        label = model.names[int(r.cls)]  # 获取类别名称
        # 获取四个角的坐标 (x1, y1, x2, y2)，保留小数点后5位
        x1, y1, x2, y2 = map(lambda v: round(v, 5), r.xyxy[0].tolist())

        # 创建字典格式的数据
        detection = {
            "label": label,
            "top_left_point": [x1, y1],
            "bottom_right_point": [x2, y2]
        }
        detections.append(detection)

    # 将检测结果写入 TXT 文件
    output_txt_file = os.path.join(output_dir, f'{img_name}.txt')
    with open(output_txt_file, 'w') as f:
        for detection in detections:
            json_line = json.dumps(detection)
            f.write(json_line + '\n')

# 批量处理目录中的图片
for img_file in os.listdir(input_dir):
    if img_file.endswith('.png'):
        image_path = os.path.join(input_dir, img_file)
        infer_and_save_labels(image_path, model, output_dir)

print(f"所有图片的检测结果已保存至 {output_dir}，带检测框的图片已自动保存在运行目录下的 'runs/detect/predict' 文件夹中。")