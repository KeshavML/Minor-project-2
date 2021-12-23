import cv2
import numpy as np
import os
import concurrent.futures as cf

os.makedirs("./dilated masks",exist_ok=True)
os.makedirs("./masked xrays",exist_ok=True)

########### DILATION ###########
def dilateMasks(mask_paths):
    global kernel
    img = cv2.imread(mask_paths[0],0)
    img_dilation = cv2.dilate(img, kernel,iterations=10)
    cv2.imwrite(mask_paths[1],img_dilation)

kernel = np.ones((5,5), np.uint8)

masks_path = "./masks/"
masks = [masks_path+each for each in os.listdir(masks_path)]
masks.sort()

save_masks = "./dilated masks/"
save_masks = [save_masks+each for each in os.listdir(masks_path)]
save_masks.sort()

images_paths = []

for idx in range(len(masks)):
    images_paths.append([masks[idx],save_masks[idx]])

with cf.ThreadPoolExecutor() as executor:
    executor.map(dilateMasks, images_paths)

del masks
del save_masks
del images_paths

########### MASKING ###########
cliplimit = 5
clahe = cv2.createCLAHE(cliplimit)

masks_path = "./dilated masks/"
masks = [masks_path+each for each in os.listdir(masks_path)]
masks.sort()

xrays_path = "./xrays/"
xrays = [xrays_path+each for each in os.listdir(xrays_path)]
xrays.sort()

save_xrays_path = "./masked xrays/"
save_xrays = [save_xrays_path+each for each in os.listdir(xrays_path)]
save_xrays.sort()

images_paths = []

for idx in range(len(masks)):
    images_paths.append([masks[idx],xrays[idx],save_xrays[idx]])

def maskXrays(paths):
    global clahe
    mask = cv2.imread(paths[0],0)
    xray = cv2.imread(paths[1],0)
    clahe = cv2.createCLAHE(cliplimit)
    masked = cv2.bitwise_and(xray, xray, mask=mask)
    masked = clahe.apply(masked)
    cv2.imwrite(paths[2],masked)

with cf.ThreadPoolExecutor() as executor:
    executor.map(maskXrays, images_paths)
# maskXrays(images_paths[0])
