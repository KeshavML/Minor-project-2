import torch
import torchvision
from DataGenerator import CovidDataset, MultiClassPathologyDataset
from torch.utils.data import DataLoader

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders_multiclass_pathology_dataset(csv_train, img_dir_train, csv_val, img_dir_val, 
        batch_size, train_transform, val_transform,
        num_workers=4, pin_memory=True):

    train_ds = MultiClassPathologyDataset(csv_file=csv_train, img_dir=img_dir_train, 
                transform=train_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers,
                pin_memory=pin_memory, shuffle=True)

    val_ds = MultiClassPathologyDataset(csv_file=csv_val, img_dir=img_dir_val,
                transform=val_transform)

    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers, 
                pin_memory=pin_memory, shuffle=False)

    return train_loader, val_loader

def get_loaders_covid_dataset(
        csv_train, img_dir_train, csv_val, img_dir_val,batch_size,
        train_transform,val_transform, num_workers=4,pin_memory=True):

    train_ds = CovidDataset(csv_file=csv_train,img_dir=img_dir_train,transform=train_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                num_workers=num_workers, pin_memory=pin_memory, shuffle=True)

    val_ds = CovidDataset(csv_file=csv_val,img_dir=img_dir_val,transform=val_transform)

    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers,
                pin_memory=pin_memory, shuffle=False)

    return train_loader, val_loader

def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_labels = 0
    # dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_labels += torch.numel(preds)
            # dice_score += (2 * (preds * y).sum()) / (
            #     (preds + y).sum() + 1e-8
            # )

    print(f"Got {num_correct}/{num_labels} with acc {num_correct/num_labels*100:.2f}")

    # print(f"Dice score: {dice_score/len(loader)}")
    model.train()

def save_predictions_as_imgs(epoch, loader, model, folder="saved_images/", device="cuda"):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()

        torchvision.utils.save_image(preds, f"{folder}/{epoch}_pred_{idx}.png")
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}/{epoch}_{idx}.png")

    model.train()
