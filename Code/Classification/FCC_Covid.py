import torch
# from utils import *
import torch.nn as nn
from configparser import ConfigParser
from torchsummary.torchsummary import summary
from utils import getModel, getLatestModel,load_checkpoint

parser = ConfigParser()
parser.read("../../Other/ConfigParser/config.ini")

# Parameters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOAD_MODEL = parser.get('CL','load_model')

class Covid(nn.Module):
    def __init__(self, init_model):
        super(Covid,self).__init__()
        self.init_model = init_model
        
        for param in self.init_model.parameters():
            param.requires_grad = False
            # print("Hello")

        self.model = nn.Sequential(
            nn.Linear(9,5),
            nn.Linear(5,1),
            nn.Sigmoid()
        )
        # return model
    def forward(self,x):
        x = self.init_model(x)
        x = self.model(x)
        return x

def getCovidModel():
    model, name = getModel()
    LOAD_MODEL_PATH = parser.get('CL','load_model_path_pathology')
    # SAVE_MODEL_PATH = parser.get('CL','load_model_path_pathology')
    # SAVE_IMAGES = parser.get('CL','save_images_pathology')+"train/"
    # SAVE_LOSS = parser.get('CL', 'save_loss_path')
    
    # SAVE_IMAGES = SAVE_IMAGES + f"{name}/"
    # SAVE_MODEL_PATH = SAVE_MODEL_PATH + f"{name}/"
    # SAVE_MODEL_COVID = SAVE_MODEL_PATH.replace('pathology','covid')
    LOAD_MODEL_PATH = LOAD_MODEL_PATH+f"{name}/"
    # LOAD_MODEL_COVID = LOAD_MODEL_PATH.replace('pathology','covid')

    if LOAD_MODEL:
        try:
            latest_model_path = getLatestModel(root=LOAD_MODEL_PATH)
            load_checkpoint(torch.load(f"{LOAD_MODEL_PATH}{latest_model_path}"), model)
        except Exception as e:
            print(f"{e}: Model couldn't be loaded.")
    # print(SAVE_MODEL_COVID)
    # print(LOAD_MODEL_COVID)
    model = Covid(model)
    return model, name

if __name__=="__main__":
    model = getCovidModel()
    summary(model)
    




