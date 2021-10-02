from albumentations.core.transforms_interface import ImageOnlyTransform
from torch import unsqueeze 
import numpy as np
import torch

class ChannelDropoutCustom(ImageOnlyTransform):
    """Randomly Drop Channels in the input Image.
    Args:
        channel_drop_range (int, int): range from which we choose the number of channels to drop.
        fill_value (int, float): pixel value for the dropped channel.
        p (float): probability of applying the transform. Default: 0.5.
    Targets:
        image
    Image types:
        uint8, uint16, unit32, float32
    """
    def apply(self, img, **params):

        # print(img[0,:,:].size())
        img = img[0,:,:].unsqueeze(0)
        return img

def main():
    img = np.random.rand(3,512,512)
    transform_ = ChannelDropoutCustom()
    img = torch.from_numpy(img)
    img = transform_.apply(img)
    print(img.size())

if __name__ == "__main__":
    main()