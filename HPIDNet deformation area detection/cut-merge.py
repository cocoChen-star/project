# import os
# import numpy as np
# import cv2 as cv
# from PIL import Image
#
#
# # 裁剪不同格式的图片时记得修改后缀
#
#
# # 裁剪图片,重复率为RepetitionRate
# def TifCrop(ImgPath, SavePath,CropSize, RepetitionRate):
#     img = cv.imread(ImgPath)
#
#     height = img.shape[0]
#     width = img.shape[1]
#
#     new_name = len(os.listdir(SavePath)) + 1
#     # 行列都是向下取整
#     for i in range(int((height - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))):
#         for j in range(int((width - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))):
#             #  如果图像是单波段
#             if (len(img.shape) == 2):
#                 cropped = img[
#                           int(i * CropSize * (1 - RepetitionRate)): int(i * CropSize * (1 - RepetitionRate)) + CropSize,
#                           int(j * CropSize * (1 - RepetitionRate)): int(j * CropSize * (1 - RepetitionRate)) + CropSize]
#             #  如果图像是多波段
#             else:
#                 cropped = img[int(i * CropSize * (1 - RepetitionRate)): int(i * CropSize * (1 - RepetitionRate)) + CropSize,
#                               int(j * CropSize * (1 - RepetitionRate)): int(j * CropSize * (1 - RepetitionRate)) + CropSize,
#                              :]
#             #  写图像
#             cv.imwrite(SavePath + "/%d.jpg" % new_name, cropped)
#             #  文件名 + 1
#             new_name = new_name + 1
# if __name__ == '__main__':
#
#     img_path = r'E:\Yolov11\YOLOv11\bailongjiang_cutting\test.jpg'
#     save_path = r'E:\Yolov11\YOLOv11\1'
#     #  将影像1裁剪为重复率为0.1的256×256的数据集
#     TifCrop(img_path, save_path, 256, 0)

# merge
import cv2 as cv
import numpy as np
import os


def TifMerge(CropPath, SavePath, OriginalSize, CropSize, RepetitionRate):
    """
    将裁剪的小图拼接回原始大图。

    Args:
        CropPath (str): 裁剪后的小图存放路径。
        SavePath (str): 拼接后的大图保存路径（需要包含文件名和扩展名）。
        OriginalSize (tuple): 原图的大小，格式为 (height, width, channels)。
        CropSize (int): 裁剪窗口的大小。
        RepetitionRate (float): 裁剪时的重叠率，范围为 [0, 1)。
    """
    # 初始化一个大画布
    height, width, channels = OriginalSize
    if channels == 1:  # 单通道图像
        merged_image = np.zeros((height, width), dtype=np.uint8)
    else:  # 多通道图像
        merged_image = np.zeros((height, width, channels), dtype=np.uint8)

    # 计算裁剪窗口的步长
    step = int(CropSize * (1 - RepetitionRate))

    # 获取所有裁剪的小图文件名
    crop_files = sorted(os.listdir(CropPath), key=lambda x: int(os.path.splitext(x)[0]))

    idx = 0
    for i in range(int((height - CropSize * RepetitionRate) / step)):
        for j in range(int((width - CropSize * RepetitionRate) / step)):
            # 读取小图
            crop_img = cv.imread(os.path.join(CropPath, crop_files[idx]))
            idx += 1

            # 计算在大图中的位置
            x_start = int(i * step)
            y_start = int(j * step)

            # 将小图放回大图
            merged_image[x_start:x_start + CropSize, y_start:y_start + CropSize] = crop_img

    # 确保文件夹存在
    os.makedirs(os.path.dirname(SavePath), exist_ok=True)

    # 保存拼接后的大图
    cv.imwrite(SavePath, merged_image)
    print(f"拼接完成，大图已保存到 {SavePath}")


# 示例调用
if __name__ == "__main__":
    CropPath = r'E:\Yolov11\YOLOv11\runs\detect\250'  # 存放裁剪后小图的文件夹路径
    SavePath = r"E:\Yolov11\YOLOv11\merge_output\merged_image.jpg"  # 拼接后的大图保存路径（包含文件名和扩展名）
    OriginalSize = (2560, 4864, 3)  # 原图的大小
    CropSize = 256  # 裁剪窗口大小
    RepetitionRate = 0  # 重叠率

    TifMerge(CropPath, SavePath, OriginalSize, CropSize, RepetitionRate)












