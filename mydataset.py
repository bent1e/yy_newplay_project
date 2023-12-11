import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np

class myDataset(Dataset):
    def __init__(self,txt_path,transform,train = False):
        self.transform = transform
        self.train = train
        self.txt_path = txt_path
        self.img_path_list = []
        for txt_ in self.txt_path:
            tmp = open(txt_).readlines()
            self.img_path_list.extend(tmp)
    def __getitem__(self, idx):
        img_item_path = self.img_path_list[idx].strip().split('\t')[0]
        img = Image.open(img_item_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        if self.train == False:
            label_pre = self.img_path_list[idx].strip().split('\t')[1]
            if label_pre =='非小玩法' or label_pre =='0':
                label = 0
            else:
                label = 1
            return img, int(label),img_item_path
        else:
            return img, 0

    def __len__(self):
        return len(self.img_path_list)