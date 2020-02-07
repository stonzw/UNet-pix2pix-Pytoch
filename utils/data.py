import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class CategoryDataset(Dataset):

    def __init__(self, root, data_list_file, trans):
        with open(os.path.join(data_list_file), 'r') as f:
            imgs = f.readlines()
        imgs = [os.path.join(root, img.strip()) for img in imgs]
        imgs = [x for x in imgs if os.path.exists(x.split(',')[0])]
        self.imgs = imgs
        self.transforms = trans

    def __getitem__(self, index):
        sample = self.imgs[index]
        splits = sample.split(',')
        img_path = splits[0]
        data = Image.open(img_path)
        data = self.transforms(data)
        label = np.int32(splits[-1])
        return data.float(), label

    def __len__(self):
        return len(self.imgs)


class PairDataset(Dataset):

    def __init__(self, root, data_list_file, trans):
        with open(os.path.join(data_list_file), 'r') as f:
            imgs = f.readlines()
        self.img_root = root
        self.imgs = imgs
        self.transforms = trans

    def __read_img(self, img_path):
        data = Image.open(img_path.strip())
        data = self.transforms(data)
        return data

    def __getitem__(self, index):
        sample = self.imgs[index]
        splits = sample.split(',')
        data1 = self.__read_img(os.path.join(self.img_root, splits[0]))
        data2 = self.__read_img(os.path.join(self.img_root, splits[1]))
        return data1.float(), data2.float()

    def __len__(self):
        return len(self.imgs)


def get_simple_preprocess(input_width_height):
    return transforms.Compose([
        transforms.Resize(input_width_height, interpolation=2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
