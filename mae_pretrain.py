import os
import argparse
import math
import torch
import torchvision
import shutil
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import ToTensor, Compose, Normalize,Resize,RandomResizedCrop,RandomHorizontalFlip
from tqdm import tqdm
torch.distributed.init_process_group(backend="nccl")
local_rank = torch.distributed.get_rank()
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
from model import *
from utils import setup_seed
from mydataset import myDataset
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--max_device_batch_size', type=int, default=24)
    parser.add_argument('--base_learning_rate', type=float, default=1.5e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--mask_ratio', type=float, default=0.75)
    parser.add_argument('--total_epoch', type=int, default=2000)
    parser.add_argument('--warmup_epoch', type=int, default=5)
    parser.add_argument('--model_path', type=str, default='/data1/zhuzhipeng/yy_newplay/checkpoints/mae/')

    args = parser.parse_args()

    setup_seed(args.seed)

    batch_size = args.batch_size
    load_batch_size = min(args.max_device_batch_size, batch_size)
    train_dataset = myDataset(['/data1/zhuzhipeng/yy_newplay/data/txt/train_v0/small_game_pos_train.txt','/data1/zhuzhipeng/yy_newplay/data/txt/train_v0/small_new_game_train_V5.0.0.txt'], transform=Compose([Resize((128,128)),RandomHorizontalFlip(p=0.3), ToTensor()]),train=True)
    dataloader = torch.utils.data.DataLoader(train_dataset, load_batch_size, sampler=DistributedSampler(train_dataset))
    print(len(train_dataset))
    writer = SummaryWriter(os.path.join('logs', 'V2.1.0_V5.0.0'))
    model = MAE_ViT(mask_ratio=args.mask_ratio).to(device)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    optim = torch.optim.AdamW(model.parameters(), lr=args.base_learning_rate * args.batch_size / 256, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    lr_func = lambda epoch: min((epoch + 1) / (args.warmup_epoch + 1e-8), 0.5 * (math.cos(epoch / args.total_epoch * math.pi) + 1))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func, verbose=True)

    accumulation_steps = 4  # 设置梯度累积的步数

    step_count = 0
    accumulative_loss = 0.0  # 记录累积的损失
    for e in range(args.total_epoch):
        model.train()
        for img, label in tqdm(iter(dataloader)):
            img = img.to(device)
            predicted_img, mask = model(img)
            loss = torch.mean((predicted_img - img) ** 2 * mask) / args.mask_ratio
            loss.backward()  # 反向传播，计算梯度
            optim.step()  # 参数更新
            optim.zero_grad()  # 梯度清零
        lr_scheduler.step()
        writer.add_scalar('mae_loss', loss.item(), global_step=e)  # 输出当前epoch的损失值
        print(f'In epoch {e}, average training loss is {loss.item()}.')
        torch.save(model.state_dict(), args.model_path + f"V2.1.0_V5.0.0/epoch_{e}.pth")