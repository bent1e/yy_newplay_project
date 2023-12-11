import os
import hnswlib
import numpy as np
import collections
import time
import pickle
import argparse
import math
import time
import torch
import torchvision
from einops import repeat, rearrange
import shutil
import torchvision.transforms.functional as F
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import ToTensor, Compose, Normalize,Resize
from tqdm import tqdm
device = torch.device("cuda:0")
from model import *
from utils import setup_seed
import torch.utils.data as Data
import torchvision.transforms as transforms
from mydataset import myDataset

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--val_batch_size', type=int, default=1)
    parser.add_argument('--model_path', type=str, default='./checkpoints/epoch_48.pth')
    parser.add_argument('--gallery_path', type=str, default=['./small_game_vals.txt'])
    parser.add_argument('--test_path', type=str, default=['/data1/zhuzhipeng/yy_newplay/MAE/new_game_test.txt'])
    args = parser.parse_args()

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    train_batch_size = args.batch_size
    val_batch_size = args.val_batch_size

    # 准备数据集和数据加载器
    val_dataset = myDataset(args.test_path, transform=transform, train=False)
    val_dataloader = Data.DataLoader(val_dataset, batch_size=val_batch_size)
    train_dataset = myDataset(args.gallery_path, transform=transform, train=False)
    train_dataloader = Data.DataLoader(train_dataset, batch_size=train_batch_size)

    # 加载模型
    model = MAE_ViT(image_size=128,
                    patch_size=4,
                    emb_dim=192,
                    encoder_layer=12,
                    encoder_head=3,
                    decoder_layer=4,
                    decoder_head=3,
                    mask_ratio=0.75).to(device)
    checkpoint_D = torch.load(args.model_path,map_location=device)
    model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint_D.items()})
    model = model.encoder
    model.eval()

    # 特征提取
    temp_result_data = []
    result_data = []
    
    print(len(train_dataloader), len(val_dataloader))
    
    for img, label, img_item_path in tqdm(iter(train_dataloader)):
        img = img.to(device)
        fea, mask = model(img)
        fea = fea[0]
        temp_result_data.append((fea.cpu().detach().numpy(), label.cpu().detach().numpy(), img_item_path))

        for _ in temp_result_data:
            for fea in range(len(_[0])):
                result_data.append((_[0][fea], _[1][fea], _[2][fea]))
        temp_result_data = []
    

    # HNSW 检索
    start_hsw = time.perf_counter()
    dim = len(result_data[0][0].flatten())
    num_elements = len(result_data)
    p = hnswlib.Index(space='l2', dim=dim)
    p.init_index(max_elements=num_elements, ef_construction=200, M=16)
    data = np.vstack([v.flatten() for v, _, _ in result_data])
    ids = np.arange(num_elements)
    p.add_items(data, ids)
    p.set_ef(50)
    correct_count = 0
    TP = 0  # True Positives
    FN = 0  # False Negatives
    f_data = []
    w_data = []
    for img, label,img_item_path in tqdm(iter(val_dataloader)):
        img = img.to(device)
        fea, mask = model(img)
        fea = fea[0]
        labels, distances = p.knn_query(fea.cpu().detach().numpy().flatten(), k=5)  # 查询最近的5个邻居
        top_labels = [result_data[idx][1] for idx in labels[0]]
        top_labels = [label.item() for label in top_labels]
        top_label = collections.Counter(top_labels).most_common(1)[0][0]
        if top_label == 0:
            crops = []
            h, w = img.shape[-2:]
            crop_size_h = h // 3
            crop_size_w = w // 3
            for i in range(3):
                for j in range(3):
                    crop = img[:, :, i * crop_size_h:(i + 1) * crop_size_h, j * crop_size_w:(j + 1) * crop_size_w]
                    crops.append(crop)
            crops = [F.resize(crop, (128, 128)) for crop in crops]
            batched_crops = torch.cat(crops, dim=0)
            with torch.no_grad():
                fea, _ = model(batched_crops)
                fea = fea[0].cpu().detach().numpy()
                for f in range(len(fea)):
                    crop_labels, _ = p.knn_query(fea[f].flatten(), k=5)  # 重新查询最近的5个邻居
                    crop_top_labels = [result_data[idx][1] for idx in crop_labels[0]]
                    crop_top_labels = [label.item() for label in crop_top_labels]
                    crop_top_label = collections.Counter(crop_top_labels).most_common(1)[0][0]
                    if crop_top_label == 1:
                        top_label =1
                        
                        break
                
        if top_label == label.item():
            correct_count += 1
            if top_label == 1:  # 如果预测为正类
                TP += 1
        else:
            if label.item() == 1:  # 如果实际为正类但预测为负类
                f_data.append(img_item_path[0])
                FN += 1
            else:
                w_data.append(img_item_path[0])
    accuracy = correct_count / len(val_dataloader)
    recall = TP / (TP + FN) if TP + FN != 0 else 0
    end_hsw = time.perf_counter()
    print(f"The accuracy is: {accuracy}")
    print(f"The recall is: {recall}")
    print(f"hsw_time is :{ end_hsw - start_hsw}")

if __name__ == '__main__':
    main()