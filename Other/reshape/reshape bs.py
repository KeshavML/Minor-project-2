import numpy as np
import os
from PIL import Image
import concurrent.futures as cf

original = './for keshav/bone suppressed/'
processed = "./for keshav/bone suppressed processed/"

if not os.path.exists(processed):
    os.makedirs(processed)

images = os.listdir(original)

def process_image(img_):
    img = Image.open(os.path.join(original,img_)).resize((512,512)).convert('I')
    img2_np  = np.asarray(img)
    Image.fromarray(np.int16((255-(img2_np/65536)*255))).convert('L').save(f'{processed}{img_}')

with cf.ThreadPoolExecutor() as executor:
    executor.map(process_image, images)
