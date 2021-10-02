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

class CovidDetectionModel(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(CovidDetectionModel, self).__init__()
        self.conv1 = conv_block(
            in_channels=in_channels,
            out_channels=32,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
        )

class Identity(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x

class conv_block(nn.Module):
    """
        Conv block for the network

        ### INPUT ###
        in_channels : in-coming feature maps (N, in_channels, W , L)
        out_channels : out-going feature maps
        kwargs : kernel size / stride / padding / other details

        ### OUTPUT ###
        output : out-going feature maps (N, out_channels, W_hat, L_hat)
    """ 
    def __init__(self, 9, 1, model, **kwargs):
        super(conv_block, self).__init__()
        # load another model
        
        self.batchnorm = nn.BatchNorm2d(9)
        self.fc_covid = nn.Linear(9, 1)

    def forward(self, x):
        x = self.batchnorm(x)
        x = self.fc_covid(x)
        return x


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


if __name__=="__main__":
    main()


# Load Data
train_dataset = CovidDataset(csv_file = "", root_dir="", transform=...)
test_dataset = CovidDataset(csv_file = "", root_dir="", transform=...)

train_loader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size,shuffle=True)
    




