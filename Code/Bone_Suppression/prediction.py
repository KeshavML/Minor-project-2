from Model import BoneSuppression
from configparser import ConfigParser
import torch
from PIL import Image
import os
import numpy as np
from torchsummary import summary
from utils import getLatestModel,load_checkpoint

parser = ConfigParser()
parser.read("../../Other/ConfigParser/config.ini")

# Parameters
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = 'cpu'
# nomalization params
MAX_PIXEL_VALUE = 255.0
MEAN = 0.449
STD = 0.226
# load images
DATASET_PATH = parser.get('BS','load_images_pred_path')
# DATASET_COVID = parser.get('BS','load_images_pred_covid')
# load model
LOAD_MODEL_PATH = parser.get('BS','load_model_path')
# save images
SAVE_IMAGES_PATH = parser.get('BS','save_images_pred_path')
# SAVE_IMAGES_COVID = parser.get('BS','save_images_pred_covid')

def save_img(img,img_name,rootdir):
    img.save(rootdir+img_name.split("/")[-1])

def predict(model, img_name):
    img = np.asarray(Image.open(img_name),dtype=np.float32)
    img = (img - MEAN*MAX_PIXEL_VALUE)/(STD * MAX_PIXEL_VALUE)
    img = torch.from_numpy(img)
    img = torch.unsqueeze(img,0)
    img = torch.unsqueeze(img,0)

    pred_img = model(img)
    pred_img = np.int16(pred_img.cpu().numpy()[0,0,:,:]*255)
    pred_img = Image.fromarray(pred_img).convert('L')
    return pred_img

if __name__ == "__main__":
########################################################            
    model = BoneSuppression(in_channels=1, out_channels=1).to(DEVICE)
    # get latest thing
    latest_model_path = getLatestModel(root=LOAD_MODEL_PATH)
    print(latest_model_path)
    load_checkpoint(torch.load(
        f"{LOAD_MODEL_PATH}{latest_model_path}", 
        map_location=DEVICE), 
        model
    )
    # summary(model)
########################################################            
    ## Predicting Pathology images
########################################################            
    DATASET = DATASET_PATH
    SAVE_IMAGES = SAVE_IMAGES_PATH
    # print(DATASET,SAVE_IMAGES)
    images = os.listdir(DATASET)
    images = [DATASET+each for each in images]
    print(images)
    with torch.no_grad():
        for each in images:
            pred_img = predict(model,each)
            save_img(pred_img,each,rootdir=SAVE_IMAGES)

########################################################            
    ## Predicting COVID images
########################################################            
    # DATASET = DATASET_COVID
    # SAVE_IMAGES = SAVE_IMAGES_COVID
    # images = os.listdir(DATASET)
    # images = [DATASET+each for each in images]
    # with torch.no_grad():
    #     for each in images:
    #         pred_img = predict(model,each)
    #         save_img(pred_img,each,rootdir=SAVE_IMAGES)
