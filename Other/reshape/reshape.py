import time
import os
import concurrent.futures as cf
from PIL import Image

original = "./original/"
processed = "./processed/"
if not os.path.exists(processed):
    os.makedirs(processed)


processes = []
img_names = os.listdir(original)
# print("aegad",img_names)

t1 = time.perf_counter()
def process_image(img_name):
    img = Image.open(f'./{original}/{img_name}').convert('L')
    img = img.resize((512, 512))
    img.save(f'./{processed}/{img_name}')
    print(f'{img_name} was processed...')

with cf.ThreadPoolExecutor() as executor:
    executor.map(process_image, img_names)

t2 = time.perf_counter()

print(f'Finished in {round(t2-t1, 3)} seconds')
