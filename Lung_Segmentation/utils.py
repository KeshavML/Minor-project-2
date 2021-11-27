import torch
import torchvision
from DataGenerator import LungSegmentationDataset
from torch.utils.data import DataLoader

def save_checkpoint(state, filename="runs/my_checkpoint.pth.tar"):
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
        
def get_loaders(train_dir, train_maskdir, val_dir,
            val_maskdir, batch_size, train_transform,
            val_transform, num_workers, pin_memory=True):

    train_ds = LungSegmentationDataset(Xray_dir=train_dir, mask_dir=train_maskdir, 
                transform=train_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers,
                pin_memory=pin_memory, shuffle=True)

    val_ds = LungSegmentationDataset(Xray_dir=val_dir, mask_dir=val_maskdir,
                transform=val_transform)

    val_loader = DataLoader(val_ds, batch_size=batch_size,  num_workers=num_workers,
                pin_memory=pin_memory, shuffle=False)

    return train_loader, val_loader

def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    # dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            # print(1,y.size()) #[batch,512,512,channels] [2,512,512,1]
            y = y[:,:,:,0]
            # print(1,y.size()) #[batch,512,512] [2,512,512]
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            # print(2,x.size()) #[batch, channel, 512, 512] [2, 1, 512, 512]
            # print(3,y.size()) #[batch, channel, 512, 512] [2, 1, 512, 512]
            # print(4,preds.size()) #[batch, channel, 512, 512] [2, 1, 512, 512]
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            # dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-5)

    print(f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}")
    # print(f"Dice score: {dice_score/len(loader)}")
    model.train()

def save_predictions_as_imgs(epoch, loader, model, folder="saved_images/", device="cpu"):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        y = y[:,:,:,0]
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()

        # predicted?
        torchvision.utils.save_image(preds, f"{folder}/{epoch}_pred_{idx}.png")
        # GT?
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}/{epoch}_{idx}.png")

    model.train()