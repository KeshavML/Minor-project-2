from DataGenerator import CovidDataset, MultiClassPathologyDataset
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
import albumentations as A
import torchvision
import datetime as dt
import torch
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

def save_checkpoint(state, root="../../OP/BS/runs/inception/"):
    """
        Saves current model state to file -> filename
    """
    print("=> Saving checkpoint")
    torch.save(state, os.path.join(root, f"{dt.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.pth.tar"))

def load_checkpoint(checkpoint, model):
    print(f"=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def getLatestModel(root):
    files = os.listdir(root)
    files.sort(reverse=True)
    return files[0]

def getModel():
    while True:
        model_name = input("Select the model(integer input):\n1) Inception\n2) ResNet34\n3) SqueezeNet\n")
        name = ''
        if model_name == '1':
            name = 'inception'
            # aux_logits = input("\nAuxilliary inputs?(Binary: 0/1)\n")
            # print()
            # if int(aux_logits) == 1:
            #     aux_logits = True
            # elif int(aux_logits) == 0:
            #     aux_logits = False
            from Inception import Inception
            model = Inception(aux_logits=False, num_classes=9)
        elif model_name == '2':
            name = 'resnet34'
            from ResNet34 import ResNet34
            model = ResNet34(img_channel=1, num_classes=9)
        elif model_name == '3':
            name = 'squeezenet'
            from SqueezeNet import squeezenet
            model = squeezenet()            
        else:
            print("\nInvalid input, try again:\n")
        if 'model' in locals():
            break
    return model, name

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
    num_pixels = 0
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
            num_pixels += torch.numel(preds)
            # dice_score += (2 * (preds * y).sum()) / (
            #     (preds + y).sum() + 1e-8
            # )

    print(f"Got {num_correct}/{num_pixels} with acc {(num_correct/num_pixels)*100}")

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
        predicted_path = f"{folder}{epoch}_pred_{idx}.png"
        torchvision.utils.save_image(preds, predicted_path)
        # GT
        gt_path = f"{folder}{epoch}_{idx}.png"
        torchvision.utils.save_image(y.unsqueeze(1), gt_path)

    model.train()
