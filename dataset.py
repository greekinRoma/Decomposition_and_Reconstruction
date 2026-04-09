from ast import parse
from Decomposition_and_Reconstruction import DeRemodel
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim  # 新增：导入优化器模块
import cv2
import os
import numpy as np
import argparse
from tqdm import tqdm
class DroneDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        """
        1. 初始化：读取路径、文件名列表等。
        """
        self.image_dir = image_dir
        self.transform = transform
        images = os.listdir(image_dir)
        self.images = [img for img in images if img.endswith('.jpg') or img.endswith('.png') or img.endswith('.jpeg')]

    def __len__(self):
        """
        2. 返回数据集的总样本量。
        """
        return len(self.images)

    def __getitem__(self, index):
        """
        3. 核心：根据索引加载一个样本。
        """
        # 获取路径
        img_path = os.path.join(self.image_dir, self.images[index])
        img_array = np.fromfile(img_path, dtype=np.uint8)
        # 2. 再用 cv2.imdecode 解码
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        image = torch.tensor(image).permute(2, 0, 1).float() / 255.0  # 转换为 PyTorch 张量并调整维度
        image = torch.unsqueeze(image, 0)  # 添加批次维度
        sample = {'img': image.cuda(), 'filename': self.images[index]}
        return sample