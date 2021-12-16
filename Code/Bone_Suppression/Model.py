# from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
import torch.nn.functional as F
import torch.nn as nn
import warnings
import torch

warnings.filterwarnings("ignore")
# writer = SummaryWriter("runs/AE")

class BoneSuppression(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(BoneSuppression, self).__init__()
        """
            Model for bone suppression.
        """
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1, bias=False),  # b, 16, 512, 512
            nn.LeakyReLU(True),
            nn.MaxPool2d(2,stride=2),  # b, 16, 256, 256

            nn.Conv2d(16, 32, 3, stride=1, padding=1, bias=False),  # b, 32, 256, 256
            nn.BatchNorm2d(32, eps=1e-05),
            nn.LeakyReLU(True),
            nn.MaxPool2d(2,stride=1),  # b, 32, 128, 128,
            
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),  # b, 64, 128, 128
            nn.BatchNorm2d(64, eps=1e-05),
            nn.LeakyReLU(True),
            nn.MaxPool2d(2,stride=2),  # b, 64, 64, 64,
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 48, 3, stride=2, padding=0),  # b, 32, 128, 128
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(48, 32, 3, stride=1, padding=1),  # b, 32, 128, 128
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(32, 24, 3, stride=2, padding=1),  # b, 16, 256, 256
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(24, 16, 3, stride=1, padding=1),  # b, 16, 256, 256
            nn.LeakyReLU(True),
            # nn.ConvTranspose2d(16, out_channels, 3, stride=2, padding=1),  # b, 1, 512, 512
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1),  # b, 1, 512, 512
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(8, out_channels, 3, stride=1, padding=1),  # b, 1, 512, 512
            # output activation function
            nn.LeakyReLU()
        )    

    def forward(self, x):
        orig = x
        x = self.encoder(x)
        x = self.decoder(x)
        x = F.interpolate(x, size=orig.shape[-1])
        return x

def test():
    x = torch.randn((3,1,512,512))
    model = BoneSuppression(in_channels=1, out_channels=1)
    # preds = model(x)
    
    summary(model, (1, 512, 512))
    ## Save model's structure
    # torch.save(model.state_dict(), './runs/Bone Suppression/Bone Suppression.pth')

    # print(preds)
    # print(x.shape)
    # assert preds.shape == x.shape, "input-output shapes do not match."

if __name__ == "__main__":
    test()



