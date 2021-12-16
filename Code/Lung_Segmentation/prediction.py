from Model import LungSegmentation
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
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# load dataset
DATASET_PATH = parser.get('LS','load_images_pred_path')
DATASET_COVID = parser.get('LS','load_images_pred_covid')
# Save dataset
LOAD_MODEL_PATH = parser.get('LS','load_model_path')
SAVE_IMAGES_PATH = parser.get('LS','save_images_pred_path')
SAVE_IMAGES_COVID = parser.get('LS','save_images_pred_covid')

MAX_PIXEL_VALUE = 255.0
MEAN = 0.449
STD = 0.226

def save_img(img,img_name,rootdir):
    img.save(rootdir+img_name.split("/")[-1])

def predict(model, img_name):
    img = np.array(Image.open(img_name),dtype=np.float32)
    img = (img - MEAN*MAX_PIXEL_VALUE)/(STD*MAX_PIXEL_VALUE)
    img = torch.from_numpy(img)
    img = torch.unsqueeze(img,0)
    img = torch.unsqueeze(img,0)

    pred_img = model(img)
    pred_img = pred_img.cpu().numpy()
    pred_img[pred_img >= 0.5] = 1.0
    pred_img[pred_img < 0.5] = 0.0
    pred_img = np.int16(pred_img[0,0,:,:]*255)
    pred_img = Image.fromarray(pred_img).convert('L')
    return pred_img

if __name__ == "__main__":
########################################################            
    model = LungSegmentation(in_channels=1, out_channels=1).to(DEVICE)
    # get latest thing
    latest_model_path = getLatestModel(root=LOAD_MODEL_PATH)
    load_checkpoint(torch.load(f"{LOAD_MODEL_PATH}{latest_model_path}"), model)

########################################################            
    ## Predicting Pathology images
########################################################            
    DATASET = DATASET_PATH
    SAVE_IMAGES = SAVE_IMAGES_PATH
    images = os.listdir(DATASET)
    images = [DATASET+each for each in images]
    with torch.no_grad():
        for each in images:
            pred_img = predict(model,each)
            save_img(pred_img,each,rootdir=SAVE_IMAGES)

########################################################            
    ## Predicting COVID images
########################################################            
    DATASET = DATASET_COVID
    SAVE_IMAGES = SAVE_IMAGES_COVID
    images = os.listdir(DATASET)
    images = [DATASET+each for each in images]
    with torch.no_grad():
        for each in images:
            pred_img = predict(model,each)
            save_img(pred_img,each,rootdir=SAVE_IMAGES)
