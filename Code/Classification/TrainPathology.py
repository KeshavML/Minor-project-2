from configparser import ConfigParser
import torch.optim as optim
from tqdm import tqdm
import torch.nn as nn
from utils import *
import torch
import gc

parser = ConfigParser()
parser.read("../../Other/ConfigParser/config.ini")

# Parameters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = True if torch.cuda.is_available() else False
LOAD_MODEL = parser.get('CL','load_model')

LEARNING_RATE = float(parser.get('CL','learning_rate'))
BATCH_SIZE = int(parser.get('CL','batch_size'))
NUM_EPOCHS = int(parser.get('CL','num_epochs'))
NUM_WORKERS = int(parser.get('CL','num_workers'))
IMAGE_HEIGHT,IMAGE_WIDTH = int(parser.get('CL','image_height')),int(parser.get('CL','image_width'))

TRAIN_IMG_DIR = parser.get('CL','train_img_dir_pathology')
TRAIN_CSV_DIR = parser.get('CL','train_csv_pathology')
VAL_IMG_DIR = parser.get('CL','val_img_dir_pathology')
VAL_CSV_DIR = parser.get('CL','val_csv_pathology')

def train_fn(loader, model, optimizer, loss_fn, scaler, SAVE_LOSS):
    loop = tqdm(loader)
    for batch_idx, (data,targets) in enumerate(loop):
        data = data['image'].to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            # print('1',data.shape)
            # print()
            # print()
            predictions = model(data)
            # print('1',predictions.shape)
            # print()
            # print()
            # print('1',type(predictions))
            # print()
            # print()
            # print('1',predictions.detach().numpy())
            # print()
            # print()
            # print('1',type(predictions.detach().numpy()))
            # print()
            # print()
            # print('1',predictions.detach().numpy()[0])
            # print()
            # print()
            targets = targets[:,0]
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
    train_transform, val_transforms = get_transforms(IMAGE_HEIGHT=IMAGE_HEIGHT,IMAGE_WIDTH=IMAGE_WIDTH)
    model, name = getModel()
    model = model.to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    LOAD_MODEL_PATH = parser.get('CL','load_model_path_pathology')
    SAVE_MODEL_PATH = parser.get('CL','load_model_path_pathology')
    SAVE_CSV = parser.get('CL','save_images_pathology')+"train/"
    SAVE_LOSS = parser.get('CL', 'save_loss_path')
    
    SAVE_CSV = SAVE_CSV + f"{name}/"
    SAVE_MODEL_PATH = SAVE_MODEL_PATH + f"{name}/"
    LOAD_MODEL_PATH = LOAD_MODEL_PATH+f"{name}/"
    SAVE_LOSS = SAVE_LOSS + f"{name}/loss.txt"

    train_loader, val_loader = get_loaders_multiclass_pathology_dataset(
        csv_train=TRAIN_CSV_DIR, img_dir_train=TRAIN_IMG_DIR, csv_val=VAL_CSV_DIR, img_dir_val=VAL_IMG_DIR,
        batch_size=BATCH_SIZE, train_transform=train_transform, val_transform=val_transforms, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    if LOAD_MODEL:
        try:
            latest_model_path = getLatestModel(root=LOAD_MODEL_PATH)
            load_checkpoint(torch.load(f"{LOAD_MODEL_PATH}{latest_model_path}"), model)
        except Exception as e:
            print(f"{e}: Model couldn't be loaded.")

    # check_loss(val_loader, model, device=DEVICE)

    scaler = torch.cuda.amp.GradScaler()
    
    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler, SAVE_LOSS)
        # save model
        checkpoint = {
            "state_dict" : model.state_dict(),
            "optimizer" : optimizer.state_dict()
        }
        if epoch % 5 == 0:
            save_checkpoint(checkpoint, root=SAVE_MODEL_PATH)
            # check_loss(val_loader, model, device=DEVICE)
            # print some examples to a folder
            # save_predictions_as_csv(epoch, val_loader, model, 
            #         folder=SAVE_CSV, device=DEVICE)
        gc.collect()

if __name__ == "__main__":
    main()

