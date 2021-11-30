from configparser import ConfigParser, ExtendedInterpolation

file = 'config.ini'

config = ConfigParser(interpolation=ExtendedInterpolation())

config['BS'] = {
    'LEARNING_RATE':'1e-4',
    'BATCH_SIZE':'2',
    'NUM_EPOCHS':'3',
    'NUM_WORKERS':'4',
    'IMAGE_HEIGHT':'512',
    'IMAGE_WIDTH':'512',
    'LOAD_MODEL':'True',
    'LOAD_MODEL_PATH':'../../OP/BS/runs/',
    'SAVE_MODEL_PATH':'../../OP/BS/runs/',
    'SAVE_IMAGES':'../../OP/BS/saved_images/',
    'TRAIN_IMG_DIR':'../../Data/BS/train/Xrays/',
    'TRAIN_MASK_DIR':'../../Data/BS/train/BSE_Xrays/',
    'VAL_IMG_DIR':'../../Data/BS/val/Xrays/',
    'VAL_MASK_DIR':'../../Data/BS/val/BSE_Xrays/',
}

config['LS'] = {
    'LEARNING_RATE':'1e-4',
    'BATCH_SIZE':'2',
    'NUM_EPOCHS':'3',
    'NUM_WORKERS':'4',
    'IMAGE_HEIGHT':'512',
    'IMAGE_WIDTH':'512',
    'LOAD_MODEL':'True',
    'LOAD_MODEL_PATH':'../../OP/LS/runs/',
    'SAVE_MODEL_PATH':'../../OP/LS/runs/',
    'SAVE_IMAGES':'../../OP/LS/saved_images/',
    'TRAIN_IMG_DIR':'../../Data/LS/train/Xrays/',
    'TRAIN_MASK_DIR':'../../Data/LS/train/Masks/',
    'VAL_IMG_DIR':'../../Data/LS/val/Xrays/',
    'VAL_MASK_DIR':'../../Data/LS/val/Masks/',
}

config['CL'] = {
    'LEARNING_RATE':'1e-4',
    'BATCH_SIZE':'2',
    'NUM_EPOCHS':'3',
    'NUM_WORKERS':'4',
    'IMAGE_HEIGHT':'512',
    'IMAGE_WIDTH':'512',
    'LOAD_MODEL':'True',
    # Pathology
    'LOAD_MODEL_PATH_PATHOLOGY':'../../OP/CL/pathology/runs/',
    'SAVE_MODEL_PATH_PATHOLOGY':'../../OP/CL/pathology/runs/',
    'SAVE_IMAGES_PATHOLOGY':'../../OP/CL/pathology/saved_images/',
    'TRAIN_IMG_DIR_PATHOLOGY':'../../Data/CL/pathology/train/xrays/',
    'TRAIN_CSV_PATHOLOGY':'../../Data/CL/pathology/train/train_pathology.csv',
    'VAL_IMG_DIR_PATHOLOGY':'../../Data/CL/pathology/val/xrays/',
    'VAL_CSV_PATHOLOGY':'../../Data/CL/pathology/val/val_pathology.csv',
    # Covid
    'LOAD_MODEL_PATH_COVID':'../../OP/CL/covid/runs/',
    'SAVE_MODEL_PATH_COVID':'../../OP/CL/covid/runs/',
    'SAVE_IMAGES_COVID':'../../OP/CL/covid/saved_images/',
    'TRAIN_IMG_DIR_COVID':'../../Data/CL/covid/train/xrays/',
    'TRAIN_CSV_COVID':'../../Data/CL/covid/train/train_covid.csv',
    'VAL_IMG_DIR_COVID':'../../Data/CL/covid/val/xrays/',
    'VAL_CSV_COVID':'../../Data/CL/covid/val/val_covid.csv',
}

with open("config.ini", 'w') as f:
    config.write(f)

