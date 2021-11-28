import torch.optim as optim
from tqdm import tqdm
import torch.nn as nn
from utils import *
import torch
import gc
import os

# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
NUM_EPOCHS = 1
NUM_WORKERS = 4
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
PIN_MEMORY = True if torch.cuda.is_available() else False
LOAD_MODEL = True
TRAIN_IMG_DIR = "./Dataset/LungPathology/Xrays/"
TRAIN_CSV = "./Dataset/LungPathology/CXR8_data.csv"
VAL_IMG_DIR = "./Dataset/LungPathology/Xrays/"
VAL_CSV = "./Dataset/LungPathology/CXR8_data.csv"

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    for batch_idx, (data,targets) in enumerate(loop):
        data = data['image'].to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            targets = targets[:,0]
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # update tqdm loop
        loop.set_postfix(loss=loss.item())
        
def main():
    train_transform, val_transforms = get_transforms(IMAGE_HEIGHT=512,IMAGE_WIDTH=512)
    model = getModel().to(DEVICE)    
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders_multiclass_pathology_dataset(
        TRAIN_CSV, TRAIN_IMG_DIR, VAL_CSV,  VAL_IMG_DIR,
        BATCH_SIZE, train_transform, val_transforms, NUM_WORKERS, PIN_MEMORY)

    if LOAD_MODEL:
        try:
            # time thingy
            models = os.listdir("../")
            models = [os.path.basename(each) for each in models]
            load_checkpoint(torch.load("../{max_time}.pth.tar"), model)
        except FileNotFoundError as e:
            print(f"{e}: Model not found.")

    # check_accuracy(val_loader, model, device=DEVICE)

    scaler = torch.cuda.amp.GradScaler()
    
    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)
        # save model
        checkpoint = {
            "state_dict" : model.state_dict(),
            "optimizer" : optimizer.state_dict()
        }
        if epoch % 5 == 0:
            save_checkpoint(checkpoint)
            check_accuracy(val_loader, model, device=DEVICE)
            # print some examples to a folder
            save_predictions_as_imgs(epoch, val_loader, model, 
                    folder="Saved Images", device=DEVICE)
        gc.collect()

if __name__ == "__main__":
    main()

