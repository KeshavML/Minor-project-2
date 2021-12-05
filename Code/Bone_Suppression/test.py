from configparser import ConfigParser

parser = ConfigParser()
parser.read("../Other/ConfigParser/config.ini")


LEARNING_RATE = parser.get('BS','learning_rate')
BATCH_SIZE = parser.get('BS','batch_size')
NUM_EPOCHS = parser.get('BS','num_epochs')
NUM_WORKERS = parser.get('BS','num_workers')
LOAD_MODEL_PATH = parser.get('BS','load_model_path')
SAVE_MODEL_PATH = parser.get('BS','save_model_path')
SAVE_IMAGES = parser.get('BS','save_images')
TRAIN_IMG_DIR = parser.get('BS','train_img_dir')
TRAIN_MASK_DIR = parser.get('BS','train_mask_dir')
VAL_IMG_DIR = parser.get('BS','val_img_dir')
VAL_MASK_DIR = parser.get('BS','val_mask_dir')


print(LEARNING_RATE)
print(BATCH_SIZE)
print(NUM_EPOCHS)
print(NUM_WORKERS)
print(LOAD_MODEL_PATH)
print(SAVE_MODEL_PATH)
print(SAVE_IMAGES)
print(TRAIN_IMG_DIR)
print(TRAIN_MASK_DIR)
print(VAL_IMG_DIR)
print(VAL_MASK_DIR)


