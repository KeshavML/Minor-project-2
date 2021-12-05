import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
from DataGenerator import CovidDataset

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Hyperparameters
# in_channel = 3
# num_classes = 10
# learning_rate = 1e-3
# batch_size = 32
# num_epochs = 1

def main():
    print("Pretrained model : ")
    print("Options: \n1) Inception\n2) ResNet\n3)SqueezeNet\n4) EfficientNet")
    model = input("Enter: ")

    # Load model
    if model == "Inception":
        model = torch.load("<model.pth>")
    elif model == "ResNet":
        model = torch.load("<model.pth>")
    elif model == "SqueezeNet":
        model = torch.load("<model.pth>")
    elif model == "EfficientNet":
        model = torch.load("<model.pth>")
    else:
        print("Wrong selection. Go again:")
        main()

    for param in model.parameters():
        param.requires_grad = False


    model = nn.Sequential([
        model,
        nn.Linear(9,5),
        nn.Linear(5,1),
    ])

    return model


if __name__=="__main__":
    main()


# Load Data
train_dataset = CovidDataset(csv_file = "", root_dir="", transform=...)
test_dataset = CovidDataset(csv_file = "", root_dir="", transform=...)

train_loader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size,shuffle=True)
    




