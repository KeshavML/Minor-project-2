# Datagenerator
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch

class CovidDataset(Dataset):
    """
        ### Parameters ###
        csv_file  : file_name = xray images' names, labels = [1x1] int list
        root_dir  : with all the xrays
        transform : transforms for augmentation

        ### Returns ###
        image, label array : (512x512, [1x1])
    """
    def __init__(self, csv_file, root_dir, transform = None):
        self.dataframe = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.file_names = self.dataframe.file_name.values.tolist()
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
        
        img_path = os.path.join(self.root_dir, self.dataframe.iloc[index].file_name) # column 1 : file_name
        image = Image.open(img_path).convert("RGB")
        y_label = torch.tensor(int(self.dataframe.iloc[index].labels)) # column 2 : labels

        if self.transform:
            image = self.transform(image)

        return image,y_label

class MultiClassPathologyDataset(Dataset):
    """
        ### Parameters ###
        csv_file  : file_name = xray images' names, labels = [1x9] int list
        folder_dir : directory with all the xrays
        transform  : transforms for augmentation

    """
    def __init__(self, csv_file, folder_dir, transform = None):
        self.dataframe = pd.read_csv(csv_file)
        self.folder_dir = folder_dir
        self.transform = transform
        self.file_names = self.dataframe.file_name.values.tolist()
        self.labels = self.dataframe.labels.values.tolist()
        self.labels = [each.split(",") for each in self.labels]

        for each in self.labels:
            for _ in each:
                _ = float(_)
        # self.labels = [each for each in self.labels]
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
        image = Image.open(os.path.join(self.folder_dir, self.file_names[index])).convert("RGB")
        label = self.labels[index]
        label = torch.tensor(np.array(label, dtype=float))
        if self.transform:
            image = self.transform(image)
            label = torch.tensor(np.array(label, dtype=float))
        
        return image, label

def test_Covid():
    dataset = CovidDataset("./Dataset/Covid/Covid_data.csv", "./Dataset/Covid/Xrays/")
    # print(dataset.file_names)
    # print(dataset.labels) 
    # print(dataset.dataframe)
    print("Length of the dataset",len(dataset))
    xray, label = dataset[0]
    print("Xray datatype : ",type(xray))
    print("Label datatype : ",type(label))
    print("Xray shape : ",xray.size)
    print("Label : ",label)

def test_Pathology():
    dataset = MultiClassPathologyDataset("./Dataset/LungPathology/CXR8_data.csv", "./Dataset/LungPathology/Xrays/")
    # print(dataset.file_names)
    # print(dataset.labels)
    # print(dataset.dataframe)
    print("Length of the dataset",len(dataset))
    xray, label = dataset[0]
    print("Xray datatype : ",type(xray))
    print("Label datatype : ",type(label))
    print("Xray shape : ",xray.size)
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