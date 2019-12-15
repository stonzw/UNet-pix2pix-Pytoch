import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from utils.data import get_simple_preprocess, PairDataset
from utils.settings import *
from models.unet import VanillaUNet


def get_UNet(*arg, **kargs):
    unet = eval(UNET_NAME)(INPUT_CHANNEL, OUTPUT_CHANNEL)
    return unet


def get_loss_func(*arg, **kargs):
    lossfunc = eval(LOSS_FUNCTION_NAME)()
    return lossfunc


def get_optimizer(*arg, **kargs):
    optimizer = eval(OPTIMIZER_NAME)(kargs['model'].parameters(), **kargs['optimizer_params'])
    return optimizer


if __name__ == '__main__':
    trans = get_simple_preprocess(INPUT_SIZE)
    train_dataset = PairDataset(IMAGE_ROOT, IMAGE_LIST_TEXT, trans)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    print(IMAGE_ROOT)
    epoch_number = EPOCH_COUNT

    loss_func = get_loss_func()
    unet = get_UNet()
    optimizer = get_optimizer(model=unet, optimizer_params=OPTIMIZER_PARAMS)
    device = torch.device(TORCH_DEVICE)

    unet.train()

    for _ in tqdm(range(epoch_number)):

        for input_img, expect_img in train_loader:
            output_img = unet(input_img)
            output_img = output_img.to(device=device, dtype=torch.float32)
            expect_img = expect_img.to(device=device, dtype=torch.float32)

            loss = loss_func(output_img, expect_img)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
