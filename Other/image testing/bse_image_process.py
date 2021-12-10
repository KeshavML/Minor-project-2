import numpy as np
from PIL import Image
import os
import concurrent.futures as cf

# for originals:
path_original = "./for keshav/original/"
original_processed_path = './for keshav/processed original/'
original_images = os.listdir(path_original)
# print(original_images)
if not os.path.exists(original_processed_path):
    os.makedirs(original_processed_path)
# works
def process_image_original(image):
    img = Image.open(os.path.join(path_original,image)).resize((512,512))
    img = 255 - np.asarray(img)
    img = Image.fromarray(img)
    img.save(f'{original_processed_path}{image}')

with cf.ThreadPoolExecutor() as executor:
    executor.map(process_image_original,original_images)

# for bses
path_bse = "./for keshav/bone suppressed/"
bse_processed_path = './for keshav/processed bse/'
bse_images = os.listdir(path_bse)
if not os.path.exists(bse_processed_path):
    os.makedirs(bse_processed_path)

def process_image_bse(image):
    img = Image.open(os.path.join(path_bse,image)).resize((512,512)).convert('I')
    img = np.int16(255-(np.asarray(img)/65532)*255)
    img = Image.fromarray(img).convert('L')
    img.save(f"{bse_processed_path}{image}")

with cf.ThreadPoolExecutor() as executor:
    executor.map(process_image_bse,bse_images)
