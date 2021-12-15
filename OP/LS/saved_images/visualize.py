from matplotlib import pyplot as plt
from PIL import Image
import numpy as np

xray_orig = np.array(Image.open("./0_0.png").convert('L'))
mask_orig = np.array(Image.open("./0_pred_0.png"))
print(mask_orig.shape)
mask = mask_orig#[:,:,0]
# print(xray.shape)
print(xray_orig.shape)
xray = xray_orig
print(xray.shape)
print(xray.max(),xray.min())

# plt.imshow(xray, interpolation='nearest')
# plt.title("Xray")
# plt.show()
plt.imshow(mask, interpolation='nearest')
plt.title("Mask")
plt.show()
