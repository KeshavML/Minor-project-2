from matplotlib import pyplot as plt
from PIL import Image
import numpy as np

xray = np.array(Image.open("./0_0.png"))[:,:,0]
mask = np.array(Image.open("./0_pred_0.png"))[:,:,0]
# print(xray.shape)
# print(sum(xray).shape)

plt.imshow(xray, interpolation='nearest')
plt.title("Xray")
plt.imshow(mask, interpolation='nearest')
plt.title("Mask")
plt.show()
