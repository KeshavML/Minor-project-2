from configparser import ConfigParser
from Model import BoneSuppression
import torch.optim as optim
from tqdm import tqdm
import torch.nn as nn
from utils import *
import torch
import gc

import warnings
warnings.filterwarnings("ignore")

parser = ConfigParser()
parser.read("../../Other/ConfigParser/config.ini")

# Parameters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = True if torch.cuda.is_available() else False
LOAD_MODEL = parser.get('BS','load_model')
SAVE_LOSS = parser.get('BS', 'save_loss')

LEARNING_RATE = float(parser.get('BS','learning_rate'))
BATCH_SIZE = int(parser.get('BS','batch_size'))
NUM_EPOCHS = int(parser.get('BS','num_epochs'))
NUM_WORKERS = int(parser.get('BS','num_workers'))

LOAD_MODEL_PATH = parser.get('BS','load_model_path')
SAVE_MODEL_PATH = parser.get('BS','save_model_path')
SAVE_IMAGES = parser.get('BS','save_images')

TRAIN_IMG_DIR = parser.get('BS','train_img_dir')
TRAIN_MASK_DIR = parser.get('BS','train_mask_dir')
VAL_IMG_DIR = parser.get('BS','val_img_dir')
VAL_MASK_DIR = parser.get('BS','val_mask_dir')

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

        write_loss(loss,filepath=SAVE_LOSS)
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
    loss_fn = nn.MSELoss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
            TRAIN_IMG_DIR, TRAIN_MASK_DIR,VAL_IMG_DIR, VAL_MASK_DIR, 
            BATCH_SIZE, train_transform, val_transform, NUM_WORKERS,PIN_MEMORY)

    if LOAD_MODEL:
        try:
            latest_model_path = getLatestModel(root=LOAD_MODEL_PATH)
            load_checkpoint(torch.load(f"{LOAD_MODEL_PATH}{latest_model_path}"), model)
        except FileNotFoundError as e:
            print(f"Error {e}: Model not found!")

    # check_loss(val_loader, model, device=DEVICE)
    
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)
        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        if epoch % 5 == 0:
            save_checkpoint(checkpoint,root=f"{SAVE_MODEL_PATH}")
            # check accuracy
            # check_loss(val_loader, model, device=DEVICE)
            # print some examples to a folder
            save_predictions_as_imgs(epoch, val_loader, model, 
                    folder=f"{SAVE_IMAGES}", device=DEVICE)
        gc.collect()

if __name__ == "__main__":
    main()
