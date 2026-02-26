from ultralytics import YOLO
# 多线程添加代码
if __name__ == '__main__':
    # 加载预训练模型
    model = YOLO('yolo11n-DySample.yaml').load('yolo11n.pt')
    # model = YOLO('yolo11n.pt')

    # 训练模型 # 现在你可以通过修改save_period的数值,就可以更改保存权重的间隔,默认值为-1(不保存)
    model.train(data='datasets.yaml',
                epochs=250,
                batch=16,
                imgsz=256,
                device=0,
                workers=0,
                save_period=5)
