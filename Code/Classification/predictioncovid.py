from configparser import ConfigParser
import torch
from torchsummary import summary
from PIL import Image
import os
import numpy as np
# from torchsummary import summary
from utils import getLatestModel,load_checkpoint,getModel

parser = ConfigParser()
parser.read("../../Other/ConfigParser/config.ini")
# Normalization params
MAX_PIXEL_VALUE = 255.0
MEAN = 0.449
STD = 0.226

def predict(model,image_name):
    img = Image.open(image_name).convert('L')
    img = np.array(img,dtype=np.float32)
    img = (img - MEAN*MAX_PIXEL_VALUE)/(STD * MAX_PIXEL_VALUE)    
    # print('shape::',img.shape)
    img = torch.from_numpy(img)
    img = torch.unsqueeze(img,0)
    img = torch.unsqueeze(img,0)
    # print('shape::',img.shape)
    pred = model(img)
    return pred

def savePreds(image_name,pred,rootdir):
    image_name = image_name.split('/')[-1]
    pred = pred.numpy()[0]
    print(pred)
    pred = np.round(pred).astype(np.uint8)
    print(pred)
    pred_labels = ''
    for each in pred:
        pred_labels=pred_labels+f',{each}'
    pred_labels = f'\"{pred_labels[1:]}\"'
    with open(rootdir,'a') as f:
        f.write(f"{image_name},{pred_labels}")
    # pass

if __name__ == "__main__":
    # Parameters
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    LOAD_MODEL = parser.get('CL','load_model')

    IMG_DIR = parser.get('CL','train_img_dir_covid')
    CSV_DIR = parser.get('CL','train_csv_covid')
    model, name = getModel()

    LOAD_MODEL_PATH = parser.get('CL','load_model_path_covid')
    SAVE_CSV = parser.get('CL','save_images_covid')+"train/"
    
    ROOTDIR_PATH = parser.get('CL','pred_csv_covid')+f'/{name}/preds.csv'

    SAVE_CSV = SAVE_CSV + f"{name}/"
    LOAD_MODEL_PATH = LOAD_MODEL_PATH+f"{name}/"

    if LOAD_MODEL:
        try:
            latest_model_path = getLatestModel(root=LOAD_MODEL_PATH)
            load_checkpoint(torch.load(f"{LOAD_MODEL_PATH}{latest_model_path}"), model)
        except Exception as e:
            print(f"{e}: Model couldn't be loaded.")

    # img = IMG_DIR+os.listdir(IMG_DIR)[0]
    with torch.no_grad():
        images = os.listdir(IMG_DIR)
        images = [IMG_DIR+each for each in images]
        for img in images:
            pred = predict(model,img)
            savePreds(img,pred,ROOTDIR_PATH)
            # print(pred)
# summary(model)





