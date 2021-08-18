
import os
import time
from glob import glob

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from data import DriveDataset
from model import build_unet
import numpy as np
from loss import DiceLoss, DiceBCELoss
from utils import seeding, create_dir, epoch_time

if __name__ == "__main__":
    """ Seeding """
    seeding(42) #42 just random number

    """ Directories """
    create_dir("files")

    """ Load dataset """
    train_x = sorted(glob("../exp_data/train/image/*"))
    train_y = sorted(glob("../exp_data/train/mask/*"))

    valid_x = sorted(glob("../exp_data/test/image/*"))
    valid_y = sorted(glob("../exp_data/test/mask/*"))


    """ Hyperparameters """
    H = 512
    W = 512
    size = (H, W)
    batch_size = 2
    num_epochs = 100
    lr = 1e-4
    checkpoint_path = "files/checkpoint.pth" #save trained model weights

    """ Dataset and loader """

    train_dataset = DriveDataset(train_x, train_y)
    train_size = len(train_dataset)
    indices = list(range(train_size))
    split = int(np.floor(0.2 * train_size))
    np.random.seed(42)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    data_str = f"Dataset Size:\nTrain: {len(train_indices)} - Valid: {len(val_indices)}\n"
    print(data_str)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=2
    )

    valid_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=2
    )

    device = torch.device('cuda')   ## GPU
    model = build_unet()
    model = model.to(device)

    summary(model, input_size=(3, 512, 512))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
    loss_fn = DiceBCELoss()

    writer = SummaryWriter()

    """ Training the model """
    best_valid_loss = float("inf")

    for epoch in range(num_epochs):
        start_time = time.time()

        epoch_loss = 0.0
        step = 0

        model.train()
        for x, y in train_loader:
            step += 1
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            optimizer.zero_grad()
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            epoch_len = len(train_dataset) // train_loader.batch_size
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)

        train_loss = epoch_loss/len(train_loader)

        epoch_loss = 0.0

        model.eval()
        with torch.no_grad():
            for x, y in valid_loader:
                x = x.to(device, dtype=torch.float32)
                y = y.to(device, dtype=torch.float32)

                y_pred = model(x)
                loss = loss_fn(y_pred, y)
                epoch_loss += loss.item()

            epoch_loss = epoch_loss/len(valid_loader)
        valid_loss = epoch_loss

        writer.add_scalar("val_loss", epoch_loss, epoch + 1)

        """ Saving the model """
        if valid_loss < best_valid_loss:
            data_str = f"Valid loss improved from {best_valid_loss:2.4f} to {valid_loss:2.4f}. Saving checkpoint: {checkpoint_path}"
            print(data_str)

            best_valid_loss = valid_loss
            torch.save(model.state_dict(), checkpoint_path)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        data_str = f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n'
        data_str += f'\tTrain Loss: {train_loss:.3f}\n'
        data_str += f'\t Val. Loss: {valid_loss:.3f}\n'
        print(data_str)

    writer.close()
