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
from dataset import DroneDataset
def get_args():
    parser = argparse.ArgumentParser(description="Road Segmentation Training")
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--train_dir', type=str, default='datasets/train', help='directory of training input images')
    parser.add_argument('--test_dir', type=str, default='datasets/test', help='directory of testing input images')
    parser.add_argument('--origin_patch_size', type=int, default=16, help='size of origin image patches')
    parser.add_argument('--origin_embed_dim', type=int, default=768, help='dimension of patch embedding')
    parser.add_argument('--resize_patch_size', type=int, default=16, help='size of resize image patches')
    parser.add_argument('--resize_embed_dim', type=int, default=768, help='dimension of patch embedding')
    parser.add_argument('--num_head',type=int, default=16, help='number of attention heads')
    parser.add_argument('--model_save_dir', type=str, default='save', help='path to save the trained model')
    parser.add_argument('--m', type=int, default=4, help='number of rows to divide the image into')
    parser.add_argument('--n', type=int, default=4, help='number of columns to divide the image into')
    return parser.parse_args()

def my_collate_fn(batch):
    images = [item for item in batch]
    return images

class IRSTDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        """
        1. 初始化：读取路径、文件名列表等。
        """
        self.image_dir = image_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

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

if __name__ == "__main__":
    args = get_args()
    dataset = IRSTDataset(image_dir=args.train_dir)
    val_dataset = DroneDataset(image_dir=args.test_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=my_collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=my_collate_fn)
    
    scale_model = DeRemodel( num_head=args.num_head, m=args.m, n=args.n, origin_patch_size=args.origin_patch_size, resize_patch_size=args.resize_patch_size, origin_embed_dim= args.origin_embed_dim, resize_embed_dim=args.resize_embed_dim).cuda()
    
    # 新增：定义 AdamW 优化器，将两个模型的参数组合在一起进行联合训练
    optimizer = optim.AdamW(
        list(scale_model.parameters()), 
        lr=args.lr
    )
    
    # 新增：将模型设置为训练模式（启用 Dropout 和 BatchNorm 的训练行为）
    scale_model.train()

    for epoch in tqdm(range(args.epochs)):
        total_loss = 0
        for i, batch in enumerate(dataloader):
            # print(f'Batch {i}') # 可以注释掉以保持 tqdm 进度条整洁
            
            # 新增：在每个 batch 开始前清空梯度
            optimizer.zero_grad() 
            
            # 将 batch 中的图像传递给分解模型
            patches = []
            attns = []
            pad_sizes = []
            orgin_sizes = []
            re_imgs = []
            
            for item in batch:
                patch, attn, pad_size, orgin_size = scale_model.encode(item['img'])
                patches.append(patch)
                attns.append(attn)
                pad_sizes.append(pad_size)
                orgin_sizes.append(orgin_size)
                
            patches = torch.concatenate(patches, dim=0)
            
            for patch, attn, pad_size, origin_size in zip(patches, attns, pad_sizes, orgin_sizes):
                patch = patch.unsqueeze(0)  # 添加批次维度
                re_img = scale_model.decode(patch=patch, attn=attn, pad_size=pad_size, orgin_size=origin_size)
                re_imgs.append(re_img)
                
            # 修改：将当前 batch 中的所有 loss 累加为 tensor，以便计算梯度
            batch_loss = 0
            for re_img, item in zip(re_imgs, batch):
                img = item['img']
                loss = torch.nn.functional.mse_loss(re_img, img)
                batch_loss = batch_loss + loss
                
            # 求该批次的平均损失
            batch_loss = batch_loss / len(batch)
            
            # 新增：反向传播计算梯度
            batch_loss.backward()
            
            # 新增：优化器更新权重参数
            optimizer.step()
            
            # 记录用于显示的总体 loss (使用 .item() 提取数值，防止显存泄漏)
            total_loss += batch_loss.item()

            scale_model.eval()  # 切换到评估模式，禁用 Dropout 和 BatchNorm 的训练行为
        with torch.no_grad():  # 禁用梯度计算，节省内存和加速推理
            for batch in val_dataloader:
                batch_loss = 0
                for item in batch:
                    img = item['img']
                    patch, attn, pad_size, orgin_size = scale_model.encode(img)
                    re_img = scale_model.decode(patch=patch, attn=attn, pad_size=pad_size, orgin_size=orgin_size)
                    loss = torch.nn.functional.mse_loss(re_img, img)
                    re_img = re_img[0].permute(1, 2, 0).cpu().numpy() * 255.0
                    re_img = re_img.astype(np.uint8)
                    save_img_path = os.path.join(args.model_save_dir,"images",f"{epoch}", item['filename'])
                    os.makedirs(os.path.dirname(save_img_path), exist_ok=True)
                    batch_loss = batch_loss + loss
                    cv2.imwrite(save_img_path, re_img)
                save_weight_path = os.path.join(args.model_save_dir,"weights", f"{epoch}.pth")
                os.makedirs(os.path.dirname(save_weight_path), exist_ok=True)
                torch.save(scale_model.state_dict(), save_weight_path)

        print(f'\nEpoch {epoch}, Train Loss: {total_loss/len(dataloader)}, Val Loss: {batch_loss/len(val_dataloader)}')