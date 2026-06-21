import numpy as np
import torch
import re


def decode_sample(sample):
    """
    自定义解码逻辑，处理 .npy 文件的图像数据和 .cls 文件的标签
    图像通道数不固定，我们根据数据形状进行动态处理
    """
    try:
        # 读取 .npy 文件（图像数据），这里我们假设数据是以字节流存储
        image = np.frombuffer(sample, dtype=np.float16)

        # 动态调整图像形状，根据通道数推测
        # 假设图像形状为 (C, W, H)，需要根据数据自动调整
        # 如果图像数据是 (N, C, W, H)，则 reshape 为 (N, C, W, H)
        # 你可以根据具体的文件维度进行处理
        image = image.reshape(-1, 72, 72)  # 这里假设你知道图像维度

        return torch.from_numpy(image.copy()).float()

    except Exception as ex:
        print(f"Error decoding sample: {ex}")
        raise

def load_npy(data):
    data = np.frombuffer(data, dtype=np.float16)
    # return torch.from_numpy(data.copy())
    return data

def augment(sample,transform):
    img = sample[0]
    if transform:
        img = transform(img)
    wavelength = sample[1]
    
    return img, wavelength


def load_mean_std(path):
    """从包含 'mean: tensor([...])' 和 'std: tensor([...])' 的txt文件中读取数据."""
    with open(path[0], "r", encoding="utf-8") as f:
        text = f.read()

    # 提取 mean 和 std 的方括号内容
    mean_str = re.search(r"mean:\s*tensor\((\[.*?\])\)", text, re.S).group(1)
    std_str  = re.search(r"std:\s*tensor\((\[.*?\])\)",  text, re.S).group(1)

    # 去掉多余空格、换行符，并解析成列表
    mean_vals = [float(x) for x in mean_str.strip("[]").replace("\n", "").split(",") if x.strip() != ""]
    std_vals  = [float(x) for x in std_str.strip("[]").replace("\n", "").split(",") if x.strip() != ""]


    return mean_vals, std_vals
