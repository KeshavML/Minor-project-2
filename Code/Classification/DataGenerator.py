from configparser import ConfigParser
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import numpy as np
import torch
import os

parser = ConfigParser()
parser.read("../../Other/ConfigParser/config.ini")
IMGAE_HEIGHT, IMAGE_WIDTH = int(parser.get('CL','image_height')),int(parser.get('CL','image_width'))

class CovidDataset(Dataset):
    """
        ### Parameters ###
        csv_file  : file_name = xray images' names, labels = [1x1] int list
        img_dir  : with all the xrays
        transform : transforms for augmentation

        ### Returns ###
        image, label array : (512x512, [1x1])
    """
    def __init__(self, csv_file, img_dir, transform = None):
        self.dataframe = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.file_names = self.dataframe.values.tolist()
        self.labels = self.dataframe.labels.values.tolist()
        
    def __len__(self):
        """
            Returns total num of xrays. (integer)
        """
        return len(self.dataframe)
    
    def __getitem__(self, index):
        """
            ### Parameters ###
            index  : acc to the csv file 

            ### Returns ###
            image, label array : (512x512, [1x1]) at that index.
        """
        # image = Image.open(os.path.join(self.img_dir, self.file_names[index]))
        img_path = os.path.join(self.img_dir, self.dataframe.iloc[index].file_name) # column 1 : file_name
        image = Image.open(img_path)#.resize((IMGAE_HEIGHT,IMAGE_WIDTH))
        image = np.expand_dims(np.array(image),-1)
        y_label = torch.tensor(np.array(int(self.dataframe.iloc[index].labels), dtype=np.float32)) # column 2 : labels

        if self.transform:
            image = self.transform(image)

        return image,y_label

class MultiClassPathologyDataset(Dataset):
    """
        ### Parameters ###
        csv_file  : file_name = xray images' names, labels = [1x9] int list
        img_dir : directory with all the xrays
        transform  : transforms for augmentation
    """
    def __init__(self, csv_file, img_dir, transform = None):
        self.dataframe = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.file_names = self.dataframe.values.tolist()
        print(f"Line 67 Columns: {self.dataframe.columns}")
        self.labels = self.dataframe['Finding Labels'].values.tolist()
        self.labels = [each.split(",") for each in self.labels]

        for each in self.labels:
            for _ in each:
                _ = float(_)

    def __len__(self):
        """
            Returns total num of xrays. (integer)
        """
        return len(self.dataframe)
    
    def __getitem__(self, index):        
        """
            ### Parameters ###
            index  : acc to the csv file 

            ### Returns ###
            image, label array : (512x512, [1x1]) at that index.
        """
        # print("-"*10)
        # print(f"{self.img_dir}{self.file_names[index][0]}")
        # print("-"*10)
        img_path = os.path.join(self.img_dir, self.file_names[index][0])
        image = Image.open(img_path)#.resize((IMGAE_HEIGHT,IMAGE_WIDTH))
        image = np.expand_dims(np.array(image),-1)
        label = self.labels[index]
        label = torch.tensor(np.array(label, dtype=np.float32))
        if self.transform:
            image = self.transform(image=image)
        return image, label
        
def test_Covid():
    dataset = CovidDataset("../../Data/CL/covid/train/train_covid.csv", "../../Data/CL/covid/train/xrays/")
    # print(dataset.file_names)
    # print(dataset.labels) 
    # print(dataset.dataframe)
    print("Length of the dataset",len(dataset))
    xray, label = dataset[0]
    print("Xray datatype : ",type(xray))
    print("Label datatype : ",type(label))
    print("Xray shape : ",xray.shape)
    print("Label : ",label)

def test_Pathology():
    dataset = MultiClassPathologyDataset("../../Data/CL/pathology/train/train_pathology.csv", "../../Data/CL/pathology/train/xrays/")
    # print(dataset.file_names)
    # print(dataset.labels)
    # print(dataset.dataframe)
    print("Length of the dataset",len(dataset))
    xray, label = dataset[0]
    # print(type(label))
    print("Xray datatype : ",type(xray))
    print("Label datatype : ",type(label))
    print("Xray shape : ",xray.shape)
    print("Label shape: ", label.size())
    print("Label : ",label)

def main():
    print("="*50)
    print("Covid Dataset test")
    print("*"*20)
    test_Covid()
    print("="*50)
    print("Lung Pathology Dataset test")
    print("*"*20)
    test_Pathology()
    print("="*50)

if __name__ == "__main__":
    main()
