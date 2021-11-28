from Bone_Suppression.Model import BoneSuppression

# from Classification.Inception import Inception
# from Classification.ResNet34 import ResNet
# from Classification.SqueezeNet import SqueezeNet

# from Lung_Segmentation.Model import LungSegmentation

from torch.utils.tensorboard import SummaryWriter

tb = SummaryWriter()

net = BoneSuppression()
tb.add_graph(model=net, verbose=True)
tb.close()
