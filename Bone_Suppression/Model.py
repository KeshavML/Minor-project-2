# import torch
import torch.nn as nn
from torchsummary import summary
# from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings("ignore")
# writer = SummaryWriter("runs/AE")
import torch.nn.functional as F

class BoneSuppression(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(BoneSuppression, self).__init__()
        """
            Model for bone suppression.
        """
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False),  # b, 16, 1024, 1024
            nn.LeakyReLU(True),
            nn.MaxPool2d(2,stride=2),  # b, 16, 512, 512

            nn.Conv2d(16, 32, 3, stride=1, padding=1, bias=False),  # b, 32, 512, 512
            nn.BatchNorm2d(32, eps=1e-05),
            nn.LeakyReLU(True),
            nn.MaxPool2d(2,stride=1),  # b, 32, 256, 256,
            
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),  # b, 64, 256, 256
            nn.BatchNorm2d(64, eps=1e-05),
            nn.LeakyReLU(True),
            nn.MaxPool2d(2,stride=2),  # b, 64, 128, 128,
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=0),  # b, 32, 256, 256
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1),  # b, 16, 512, 512
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1),  # b, 1, 1024, 1024
            # output activation function
            nn.Tanh()
        )    

    def forward(self, x):
        orig = x
        x = self.encoder(x)
        x = self.decoder(x)
        x = F.interpolate(x, size=orig.shape[-1])
        return x

def test():
    # x = torch.randn((3,1,512,512))
    model = BoneSuppression(in_channels=1,out_channels=1)
    # preds = model(x)
    
    print(summary(model, (1, 512, 512)))
    ## Save model's structure
    # torch.save(model.state_dict(), './runs/Bone Suppression/Bone Suppression.pth')

    # print(preds.shape)
    # print(x.shape)
    # assert preds.shape == x.shape, "input-output shapes do not match."

if __name__ == "__main__":
    test()



