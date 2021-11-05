import torch
import torchvision
import numpy as np
from DataGenerator import BoneSuppressionDataset
from torch.utils.data import DataLoader

def save_checkpoint(state, filename="./runs/my_checkpoint.pth.tar"):
    """
        Saves current model state to file -> filename
    """
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    """
        Loads existing model
    """
    try:
        model.load_state_dict(checkpoint["state_dict"])
        print("=> Loading checkpoint")
    except FileNotFoundError as e:
        print(f"{e}: Model not found.")

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

    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers,
                pin_memory=pin_memory,shuffle=True)

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
    print(f"MSE score: {mse_score/len(loader)}")
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