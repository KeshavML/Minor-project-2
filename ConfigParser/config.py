from configparser import ConfigParser, ExtendedInterpolation

file = 'config.ini'

config = ConfigParser(interpolation=ExtendedInterpolation())

config['bone_suppression'] = {
    'LEARNING_RATE':'1e-4',
    'BATCH_SIZE':'4',
    'NUM_EPOCHS':'50',
    'NUM_WORKERS':'2',
    'IMAGE_HEIGHT':'512',
    'IMAGE_WIDTH':'512',
    'LOAD_MODEL':'True',
    'TRAIN_IMG_DIR':'Dataset/BSE_Xrays/',
    'TRAIN_MASK_DIR':'Dataset/Xrays/',
    'VAL_IMG_DIR':'Dataset/BSE_Xrays/',
    'VAL_MASK_DIR':'Dataset/Xrays/',
}

config['lung_segmentation'] = {
    'LEARNING_RATE':'1e-4',
    'BATCH_SIZE':'1',
    'NUM_EPOCHS':'5',
    'NUM_WORKERS':'4',
    'IMAGE_HEIGHT':'512',
    'IMAGE_WIDTH':'512',
    'PIN_MEMORY':'False',
    'LOAD_MODEL':'True',
    'TRAIN_IMG_DIR':'./Dataset/Training/Xrays/',
    'TRAIN_MASK_DIR':'./Dataset/Training/Masks/',
    'VAL_IMG_DIR':'./Dataset/Validation/Xrays/',
    'VAL_MASK_DIR':'./Dataset/Validation/Masks/',
}

config['classification'] = {}





# print(config['files']['python_path'])
# with open(file, 'w') as f:
#     config.write(f)

