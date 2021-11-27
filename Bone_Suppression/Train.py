from Model import BoneSuppression
import torch.optim as optim
from tqdm import tqdm
import torch.nn as nn
from utils import *
import torch
import gc

# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2
NUM_EPOCHS = 2
NUM_WORKERS = 4
# IMAGE_HEIGHT = 512  # 1280 originally
# IMAGE_WIDTH = 512  # 1918 originally
PIN_MEMORY = True if torch.cuda.is_available() else False
LOAD_MODEL = True
TRAIN_IMG_DIR = "Dataset/BSE_Xrays/"
TRAIN_MASK_DIR = "Dataset/Xrays/"
VAL_IMG_DIR = "Dataset/BSE_Xrays/"
VAL_MASK_DIR = "Dataset/Xrays/"

def train_fn(loader, model, optimizer, loss_fn, scaler):
    """
       Code for 1 epoch 
    """
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            targets = targets[:,:,:,:,0]
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())

def main():
    train_transform, val_transform = get_transforms()

    model = BoneSuppression(in_channels=1, out_channels=1).to(DEVICE)
    # Change the loss function
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training and validation data loaders
    train_loader, val_loader = get_loaders(
            TRAIN_IMG_DIR, TRAIN_MASK_DIR,
            VAL_IMG_DIR, VAL_MASK_DIR, BATCH_SIZE,
            train_transform, val_transform, NUM_WORKERS,PIN_MEMORY)

    # Load if model exists
    if LOAD_MODEL:
        try:
            load_checkpoint(torch.load("./runs/my_checkpoint.pth.tar"), model)
        except Exception as e:
            print(f"Error {e}: Model not found!")

    check_accuracy(val_loader, model, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()

    # Training loop
    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        if epoch % 5 == 0:
            save_checkpoint(checkpoint)
            # check accuracy
            check_accuracy(val_loader, model, device=DEVICE)
            # print some examples to a folder
            save_predictions_as_imgs(epoch, val_loader, model, 
                    folder="saved_images/", device=DEVICE)
        gc.collect()

if __name__ == "__main__":
    main()