from albumentations.core.transforms_interface import ImageOnlyTransform

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
        print(img[0,:,:].size())
        return img[0,:,:]
