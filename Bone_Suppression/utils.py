from DataGenerator import BoneSuppressionDataset
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
import albumentations as A
import datetime as dt
import torchvision
import numpy as np
import torch
# import glob
import os

# Index:
# 1) save_checkpoint
# 2) load_checkpoint
# 3) get_transforms
# 4) get_loaders
# 5) check_accuracy
# 6) save_predictions_as_imgs

def save_checkpoint(state, root="../../OP/BS/runs/"):
    """
        Saves current model state to file -> filename
    """
    print("=> Saving checkpoint")
    torch.save(state, os.path.join(root, f"{dt.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.pth.tar"))

def load_checkpoint(checkpoint, model):
    """
        Loads existing model
    """
    try:
        model.load_state_dict(checkpoint["state_dict"])
        print("=> Loading checkpoint")
    except Exception as e:
        print(f"{e}: Model couldn't be loaded.")

def getLatestModel(root):
    files = os.listdir(root)
    files.sort(reverse=True)
    return files[0]

def get_transforms():
    """
        Image augmentations and transforms.

        ### Returns ###
        train_transform and val_transform
    """
    train_transform = A.Compose([
            A.Rotate(limit=10, p=0.4), A.HorizontalFlip(p=0.5),
            A.Normalize(mean=0.0, std=1.0, max_pixel_value=255.0,), ToTensorV2()])
    
    val_transform = A.Compose([
            A.Normalize(mean=0.0, std=1.0, max_pixel_value=255.0,), ToTensorV2()])
    
    return train_transform, val_transform

def get_loaders( train_dir, train_maskdir, val_dir, val_maskdir, batch_size, 
            train_transform, val_transform, num_workers=4, pin_memory=True):
    """
        ### Input ###
        train_dir       : directory with training Xray images
        train_maskdir   : directory with training BSE Xray images
        val_dir         : directory with validation Xray images
        val_maskdir     : directory with validation BSE Xray images
        batch_size      : batch_size (4,8,16,32)
        train_transform : transforms for data augmentation for training data
        val_transform   : transforms for data augmentation for validation data
        num_workers     : processor cores for data loading
        pin_memory      : This lets your DataLoader allocate the samples in page-locked memory, which speeds-up the transfer.

        ### Returns ###
        train_loader    : Training dataloader
        val_loader      : Validation dataloader
    """
    train_ds = BoneSuppressionDataset(Xray_dir=train_dir, mask_dir=train_maskdir,
                transform=train_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers,pin_memory=pin_memory,shuffle=True)

    val_ds = BoneSuppressionDataset(Xray_dir=val_dir, mask_dir=val_maskdir,
                transform=val_transform)

    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers,
                pin_memory=pin_memory,shuffle=False)

    return train_loader, val_loader

def check_accuracy(loader, model, device="cuda"):
    """
        This function provides an evaluation score.

        ### Input ###
        loader      : Data loader
        model       : Bone Suppression model
        device      : CPU/CUDA (GPU)

        ### Returns ###

    """
    num_correct = 0
    num_pixels = 0
    mse_score = 0
    
    # To turn off batchnorm, dropouts.
    model.eval()

    # To turn off gradient calculation.
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y[:,:,:,0]
            y = y.to(device).unsqueeze(1)
            preds = model(x)
            mse_score += np.square(np.subtract(y,preds)).mean()

    # print(f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}")
    # print(f"MSE score: {mse_score/len(loader)}")
    model.train()

def save_predictions_as_imgs(epoch, loader, model, folder="saved_images/", device="cuda"):
    """
        This function terrifies me.
    """
    model.eval()
    for idx, (x, y) in enumerate(loader):
        y = y[:,:,:,0]
        x = x.to(device=device)
        with torch.no_grad():
            preds = model(x)
        # what image is this? Predicted?
        torchvision.utils.save_image(preds, f"{folder}/{epoch}_pred_{idx}.png")
        # and what image is this mate? GT?
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}/{epoch}_{idx}.png")

    model.train()