from torchsummary import summary
from torch import nn
import warnings
import torch

warnings.filterwarnings("ignore")

class Inception(nn.Module):
    """
        Inception V1/GoogleNet based model (because fewer inception layers)

        ### INPUT ###
        aux_logits : Auxiliary outputs? (Binary) Default = True
        num_classes : Number of classes  [pathologies + normal] (Integer)

        ### CONSTANTS ###
        in_channels : channels of images = 1 (integer)

        ### OUTPUT ###
        IF Aux_logits:
            aux1, aux1, x (shape for all : [batch_size x num_classes])
        ELSE:
            x : shape : [batch_size x num_classes]
    """

    def __init__(self, aux_logits=False, num_classes=9):
        super(Inception, self).__init__()
        assert aux_logits == True or aux_logits == False
        self.aux_logits = aux_logits

        # Write in_channels, etc, all explicit in self.conv1, rest will write to
        # make everything as compact as possible, kernel_size=3 instead of (3,3)
        self.conv1 = conv_block(
            in_channels=1,
            out_channels=32,
            kernel_size=7,
            stride=2,
            padding=3,
        )

        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = conv_block(32, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # In this order: in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool
        self.inception3a = Inception_block(64, 26, 32, 46, 16, 20, 20)  #
        # self.inception3b = Inception_block(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception4a = Inception_block(112, 51, 51, 77, 13, 38, 26) #
        self.inception4b = Inception_block(192, 102, 44, 110, 8, 24, 32) # 
        # self.inception4c = Inception_block(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception_block(268, 110, 90, 180, 20, 40, 40)
        # self.inception4e = Inception_block(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception5a = Inception_block(370, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception_block(832, 250, 190, 100, 48, 20, 64)
        self.maxpool5 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.inception6a = Inception_block(434, 160, 120, 64, 32, 16, 32)
        
        self.avgpool = nn.AvgPool2d(kernel_size=7,padding=1, stride=1)
        self.dropout1 = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(4352, 512)
        self.dropout2 = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(512,128)
        self.dropout3 = nn.Dropout(p=0.3)
        self.fc3 = nn.Linear(128,num_classes)

        if self.aux_logits:
            self.aux1 = InceptionAux(192, num_classes)
            self.aux2 = InceptionAux(370, num_classes)
        else:
            self.aux1 = self.aux2 = None

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        # x = self.conv3(x)
        x = self.maxpool2(x)

        x = self.inception3a(x)
        # x = self.inception3b(x)
        x = self.maxpool3(x)

        x = self.inception4a(x)

        # Auxiliary Softmax classifier 1
        if self.aux_logits and self.training:
            aux1 = self.aux1(x)

        x = self.inception4b(x)
        # x = self.inception4c(x)
        x = self.inception4d(x)

        # Auxiliary Softmax classifier 2
        if self.aux_logits and self.training:
            aux2 = self.aux2(x)

        # x = self.inception4e(x)
        x = self.maxpool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.maxpool5(x)
        x = self.inception6a(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.dropout3(x)
        x = self.fc3(x)


        if self.aux_logits and self.training:
            return aux1, aux2, x
        else:
            return x


class Inception_block(nn.Module):
    """
        Inception Block with 4 branches (can modify to inception V3 here)

        ### INPUT ###
        in_channels : input feature maps (N, C, W, L)
        
        out_1x1 : output channels for 1x1 conv (N, out_1x1, W_hat, L_hat)
        
        red_3x3 : Reduction in channels for next layer
        out_3x3 : output channels for 3x3 conv (N, out_3x3, W_hat, L_hat)
        
        red_5x5 : Reduction in channels for next layer
        out_5x5 : output channels for 5x5 conv (N, out_5x5, W_hat, L_hat)
        
        out_1x1pool : output channels for 1x1 maxpool-conv (N, out_1x1pool, W_hat, L_hat)

        ### OUTPUT ###
        Concatenated feature maps : [N, Channels, W_hat, L_hat]
                # Channels = out_1x1+out_3x3+out_5x5+out_1x1pool

    """
    def __init__(self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool):
        super(Inception_block, self).__init__()
        self.branch1 = conv_block(in_channels, out_1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            conv_block(in_channels, red_3x3, kernel_size=1),
            conv_block(red_3x3, out_3x3, kernel_size=3, padding=1),
        )

        self.branch3 = nn.Sequential(
            conv_block(in_channels, red_5x5, kernel_size=1),
            conv_block(red_5x5, out_5x5, kernel_size=5, padding=2),
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            conv_block(in_channels, out_1x1pool, kernel_size=1),
        )

    def forward(self, x):
        a = self.branch1(x)
        b = self.branch2(x)
        c = self.branch3(x)
        d = self.branch4(x)
        return torch.cat(
            [a, b, c, d], 1
        )


class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        """
            Aux output node (will may be use for covid???)
            ### INPUT ###
            in_channels : in-coming feature maps (N, in_channels, W , L)
            num_classes : num of label classes

            ### OUTPUT ###
            labels = [1xnum_classes]            

        """
        super(InceptionAux, self).__init__()
        self.relu = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(p=0.5)
        self.pool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = conv_block(in_channels, 128, kernel_size=1)
        self.fc1 = nn.Linear(12800, 1024)
        self.fc2 = nn.Linear(1024, 420)
        self.dropout2 = nn.Dropout(p=0.4)
        self.fc3 = nn.Linear(420,128)
        self.dropout3 = nn.Dropout(p=0.4)
        self.fc4 = nn.Linear(128,num_classes)

    def forward(self, x):
        """
            Regular forward function for Aux outputs

            ### RETURNS ###
            labels = [1xnum_classes]
        """
        x = self.pool(x)
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.dropout3(x)
        x = self.fc4(x)

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
    def __init__(self, in_channels, out_channels, **kwargs):
        super(conv_block, self).__init__()
        self.relu = nn.LeakyReLU()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.relu(self.batchnorm(self.conv(x)))


if __name__ == "__main__":
    # N = 3 (Mini batch size)
    x = torch.randn(3, 1, 512, 512)
    # output = aux1, aux2, x if aux_logits == True else x
    model = Inception(aux_logits=True, num_classes=9)
    print(type(model(x)))
    # summary(model, (1, 512, 512))
    # print(model)

    # ## Save model's structure
    # torch.save(model.state_dict(), './runs/Classification/Inception/Inception.pth')
