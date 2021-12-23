from torchsummary import summary
import torch.nn as nn
import warnings
import torch

warnings.filterwarnings("ignore")

class block(nn.Module):
    def __init__(self, in_channels, intermediate_channels, identity_downsample=None, stride=1):
        super(block, self).__init__()
        self.expansion = 2 
        self.conv1 = nn.Conv2d(
            in_channels, 
            intermediate_channels, 
            kernel_size=1, 
            stride=1, 
            padding=0, 
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(intermediate_channels)
        self.conv2 = nn.Conv2d(
            intermediate_channels, 
            intermediate_channels, 
            kernel_size=3,
            stride=stride, 
            padding=1, 
            bias=False
        )

        self.bn2 = nn.BatchNorm2d(intermediate_channels)
        self.conv3 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels * self.expansion,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )

        self.bn3 = nn.BatchNorm2d(intermediate_channels * self.expansion)
        self.relu = nn.LeakyReLU()
        self.identity_downsample = identity_downsample
        self.stride = stride

    def forward(self, x):
        identity = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x

class ResNet(nn.Module):
    def __init__(self, block, layers, image_channels, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 32
        self.conv1 = nn.Conv2d(image_channels, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.LeakyReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Essentially the entire ResNet architecture are in these 4 lines below
        self.layer1 = self._make_layer(block, layers[0], intermediate_channels=32, stride=1)
        self.layer2 = self._make_layer(block, layers[1], intermediate_channels=64, stride=2)        
        self.layer3 = self._make_layer(block, layers[2], intermediate_channels=128, stride=2)        
        self.layer4 = self._make_layer(block, layers[3], intermediate_channels=256, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(256 * 2, 128)
        
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x

    def _make_layer(self, block, num_residual_blocks, intermediate_channels, stride):
        identity_downsample = None
        layers = []

        # Either if we half the input space for ex, 56x56 -> 28x28 (stride=2), or channels changes
        # we need to adapt the Identity (skip connection) so it will be able to be added
        # to the layer that's ahead
        if stride != 1 or self.in_channels != intermediate_channels * 2:
            identity_downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    intermediate_channels * 2,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(intermediate_channels * 2),
            )

        layers.append(
            block(self.in_channels, intermediate_channels, identity_downsample, stride)
        )

        # The expansion size is 2 for this ResNet based network 
        self.in_channels = intermediate_channels * 2

        # For example for first resnet layer: 256 will be mapped to 128 as intermediate layer,
        # then finally back to 256. Hence no identity downsample is needed, since stride = 1,
        # and also same amount of channels.
        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, intermediate_channels))

        return nn.Sequential(*layers)


def ResNet34(img_channel=1, num_classes=9):
    return ResNet(block, [2, 3, 4, 2], img_channel, num_classes) 

def test():
    model = ResNet34(img_channel=1, num_classes=9)
    # model2 = ResNet34(img_channel=2, num_classes=9)
    # model3 = ResNet34(img_channel=3, num_classes=9)
    # y = model(torch.randn(3, 1, 512, 512))#.to("cuda")
    # print(y.size())
    # print(model)
    summary(model, (1, 512, 512))
    # x1 = torch.randn(3, 1, 512, 512)
    # x2 = torch.randn(3, 2, 512, 512)
    # x3 = torch.randn(3, 3, 512, 512)
    # print("Ouput 1: ",type(model1(x1)))
    # print("Ouput 2: ",model2(x2).shape)
    # print("Ouput 3: ",type(model3(x3)))

    ## Save model's structure
    # torch.save(model.state_dict(), './runs/Classification/ResNet/ResNet.pth')

if __name__=="__main__":
    test()
