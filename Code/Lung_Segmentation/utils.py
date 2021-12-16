from DataGenerator import LungSegmentationDataset
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

def save_checkpoint(state, root="../../OP/LS/runs/"):
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
    except FileNotFoundError as e:
        print(f"{e}: Model not found.")
        
def getLatestModel(root):
    files = os.listdir(root)
    files.sort(reverse=True)
    return files[0]

def get_transforms():
    train_transform = A.Compose([
        A.Rotate(limit=35, p=1.0), A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.1),
        A.Normalize(mean=[0.449], std=[0.226], max_pixel_value=255.0),
        ToTensorV2()])

    val_transform = A.Compose([
        A.Normalize(mean=[0.449], std=[0.226], max_pixel_value=255.0),
        ToTensorV2()])

    return train_transform, val_transform


def get_loaders(train_dir, train_maskdir, val_dir,
            val_maskdir, batch_size, train_transform,
            val_transform, num_workers, pin_memory=True):

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

    train_ds = LungSegmentationDataset(Xray_dir=train_dir, mask_dir=train_maskdir, 
                transform=train_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers,
                pin_memory=pin_memory, shuffle=True)

    val_ds = LungSegmentationDataset(Xray_dir=val_dir, mask_dir=val_maskdir,
                transform=val_transform)

    val_loader = DataLoader(val_ds, batch_size=batch_size,  num_workers=num_workers,
                pin_memory=pin_memory, shuffle=False)

    return train_loader, val_loader

def write_loss(loss_val, filepath='../../OP/LS/runs/loss.txt'):
    print("*"*50)
    loss_val = str(round(loss_val.item(),4))
    print(f"Loss val : {loss_val}")
    data = f"{dt.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')},{loss_val}"
    with open(filepath,'a') as f:
        f.write(data)
        f.write("\n")
    print("*"*50)

# def check_loss(loader, model, device="cuda"):
#     num_correct = 0
#     num_pixels = 0
#     dice_score = 0
#     model.eval()

#     with torch.no_grad():
#         for x, y in loader:
#             x = x.to(device)
#             y = y[:,:,:,0]
#             y = y.to(device).unsqueeze(1)
#             preds = torch.sigmoid(model(x))
#             preds = (preds > 0.5).float()
#             num_correct += (preds == y).sum()
#             num_pixels += torch.numel(preds)
#             dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-5)

#         print(num_correct) 
#         print(num_pixels)
#         print(f"Got {num_correct}/{num_pixels} with acc {(num_correct/num_pixels)*100:.3f}%")
#     print(f"Dice score: {dice_score/len(loader)}")
#     model.train()

def save_predictions_as_imgs(epoch, loader, model, folder="../../OP/LS/saved_images/", device="cpu"):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        # print(y.shape)
        # print("$"*40)
        y = y[0,:,:,0]
        # print('1',y.shape)
        # print('2',y.max(),y.min())
        y[y == 1] = 255
        # print('3',y.shape)
        # print('4',y.max(),y.min())
        x = x.to(device=device)
        # print('5',x.shape)
        # print('6',x.max(),x.min())
        
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
        #     # preds = (preds > 0.5).float()
        # print('7',preds.shape)
        # print('8',preds.max(),preds.min())
        # print("$"*40)

        # # predicted
        preds = preds.cpu().numpy()[0,0,:,:]
        preds[preds<0.5] = 0.0
        preds[preds>=0.5] = 1.0
        preds = np.int16(preds*255)
        preds = Image.fromarray(preds).convert('L')
        # preds.show()
        # preds.save(f"{folder}/{epoch}_pred_{idx}.png")

        # # GT
        y = y.cpu().numpy()
        print('unique:',np.unique(y))
        # print('sdfas',y.max())
        y = np.int16(y)
        # print(y.shape)
        # print('unique:',np.unique(y))
        # print(y.max())
        y = Image.fromarray(y).convert('L')
        # y.show()
        y.save(f"{folder}/{epoch}_{idx}.png")
        # torchvision.utils.save_image(y[0,:,:], f"{folder}/{epoch}_{idx}.png")

    model.train()
