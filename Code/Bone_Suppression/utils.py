from DataGenerator import BoneSuppressionDataset
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from PIL import Image
import albumentations as A
import datetime as dt
import torchvision
import numpy as np
import torch
import os

# Index:
# 1) save_checkpoint
# 2) load_checkpoint
# 3) getLatestModel
# 4) get_transforms
# 5) get_loaders
# 6) check_accuracy
# 7) save_predictions_as_imgs

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
    try:
        return files[0]
    except IndexError:
        print("Model not found. Creating new one.")

def get_transforms():
    """
        Image augmentations and transforms.

        ### Returns ###
        train_transform and val_transform
    """
    train_transform = A.Compose([
            A.Rotate(limit=10, p=0.4), A.HorizontalFlip(p=0.5),
            A.Normalize(mean=0.449, std=0.226, max_pixel_value=255.0,), ToTensorV2()])
    
    val_transform = A.Compose([
            A.Normalize(mean=0.449, std=0.226, max_pixel_value=255.0,), ToTensorV2()])
    
    return train_transform, val_transform

def get_loaders(train_dir, train_maskdir, val_dir, val_maskdir, batch_size, 
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

    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers,
                pin_memory=pin_memory,shuffle=True)

    val_ds = BoneSuppressionDataset(Xray_dir=val_dir, mask_dir=val_maskdir,
                transform=val_transform)

    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers,
                pin_memory=pin_memory,shuffle=False)

    return train_loader, val_loader

def write_loss(loss_val, filepath='../../OP/BS/loss.txt'):
    print("*"*50)
    loss_val = str(round(loss_val.item(),4))
    print(f"Loss val : {loss_val}")
    data = f"{dt.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')},{loss_val}"
    with open(filepath,'a') as f:
        f.write(data)
        f.write("\n")
    print("*"*50)

# def check_loss(loader, model, device="cuda"):
#     """
#         This function provides an evaluation score.

#         ### Input ###
#         loader      : Data loader
#         model       : Bone Suppression model
#         device      : CPU/CUDA (GPU)

#         ### Returns ###

#     """
#     num_correct = 0
#     num_pixels = 0
#     mse_score = 0
#     print("This is running")
    
#     # To turn off batchnorm, dropouts.
#     model.eval()

#     # To turn off gradient calculation.
#     with torch.no_grad():
#         for x, y in loader:
#             x = x.to(device)
#             y = y[:,:,:,0]
#             y = y.to(device).unsqueeze(1)
#             preds = model(x)
#             mse_score += np.square(np.subtract(y,preds)).mean()

#     # print(f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}")
#     # print(f"MSE score: {mse_score/len(loader)}")
#     model.train()

def save_predictions_as_imgs(epoch, loader, model, folder="../../OP/BS/saved_images/", device="cuda"):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        y = y[:,:,:,0].cpu()
        # Image
        y = np.int16(y[0,:,:].numpy())
        # print("y",y.max(),y.min())
        y = Image.fromarray(y).convert("L")
        # print("gt",np.array(y).max(),np.array(y).min())
        # y.show()
        # y.save(f"{folder}/{epoch}_{idx}.png")

        x = x.to(device=device)
        # print(x)
        # print(dir(x))
        # print(type(x))
        # print(x.shape)
        with torch.no_grad():
            preds = model(x)
        # print("preds",preds.max(),preds.min())
        preds = np.int16(preds.cpu().numpy()[0,0,:,:]*255)
        preds = Image.fromarray(preds).convert("L")
        # print('preds2',np.absolute(np.array(preds)).max(),np.absolute(np.array(preds)).min())
        preds.save(f"{folder}/{epoch}_pred_{idx}.png")
        # preds.show()

    model.train()
