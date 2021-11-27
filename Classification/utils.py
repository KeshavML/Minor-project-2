import torch
import torchvision
from DataGenerator import CovidDataset, MultiClassPathologyDataset
from torch.utils.data import DataLoader
import datetime
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os

# Index:
# 1) save_checkpoint
# 2) load_checkpoint
# 3) getModel
# 4) get_transforms
# 5) get_loaders_multiclass_pathology_dataset
# 6) get_loaders_covid_dataset
# 7) check_accuracy
# 8) save_predictions_as_imgs

def save_checkpoint(state, filename="../"):
    now = datetime.now()
    current_time = now.strftime("%H_%M_%S")
    file_path = os.path.join(filename,current_time)
    if not os.path.isdir(file_path):
        os.makedirs(file_path)
    print(f"=> Saving checkpoint: {current_time}")
    torch.save(state, file_path+f"/{current_time}.pth.tar")

def load_checkpoint(checkpoint, model):
    print(f"=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def getModel():
    while True:
        model_name = input("Select the model(integer input):\n1) Inception\n2) ResNet34\n3) SqueezeNet\n")
        if model_name == '1':
            aux_logits = input("\nAuxilliary inputs?(Boolean)\n")
            print()
            if aux_logits:
                aux_logits = True
            else:
                aux_logits = False
            from Inception import Inception
            model = Inception(aux_logits=aux_logits, num_classes=9)
        elif model_name == '2':
            from ResNet34 import ResNet34
            model = ResNet34(img_channel=1, num_classes=9)
        elif model_name == '3':
            from SqueezeNet import squeezenet
            model = squeezenet()            
        else:
            print("\nInvalid input, try again:\n")
        if 'model' in locals():
            break
    return model

def get_transforms(IMAGE_HEIGHT, IMAGE_WIDTH):
    train_transform = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Rotate(limit=35, p=1.0), A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.1),
        A.Normalize(mean=[0.0], std=[1.0], max_pixel_value=255.0),
        ToTensorV2()])

    val_transform = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width =IMAGE_WIDTH),
        A.Normalize(mean=0.0, std=[1.0], max_pixel_value=255.0),
        ToTensorV2()])

    return train_transform, val_transform

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
            x = x['image'].to(device)
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

def save_predictions_as_imgs(epoch, loader, model, folder="Saved Images", device="cuda"):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x['image'].to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()

        # Predicted
        torchvision.utils.save_image(preds, f"{folder}/{epoch}_pred_{idx}.png")
        # GT
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}/{epoch}_{idx}.png")

    model.train()
