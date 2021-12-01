import os
import configparser as cp
"""
PROJ_PATH = '/Desktop/project_notes/Minor/'

# os.mkdir(path, mode=0o777, directory_fd[optional])

# read the config parser file, and create the directory structure according to the variables

# for each section in configuration file
	# make a constant list of options we need
	# create directories by accessing the tuples in configuration file with cp.section[option] = value syntax
# done

config = cp.ConfigParser()
config.read('config.ini')

BS_PATH = [
	'load_model_path', 'save_images', 'train_img_dir', 'train_mask_dir', 'val_img_dir', 'val_mask_dir'
]

LS_PATH = [
	'load_model_path', 'save_images', 'train_img_dir', 'train_mask_dir', 'val_img_dir', 'val_mask_dir'
]

CL_PATH = [
	'load_model_path_pathology', 'save_images_pathology', 'train_img_dir_pathology', 'val_img_dir_pathology', 
	'load_model_path_covid', 'save_images_covid', 'train_img_dir_covid', 'val_img_dir_covid'
]

bs_dirs, ls_dirs, cl_dirs = [], [], []

for section in config.sections():
	for option in config[section]:
		if option in BS_PATH and section == 'BS':
			bs_dirs.append(config[section][option])
		if option in LS_PATH and section == 'LS':
			ls_dirs.append(config[section][option])
		if option in CL_PATH and section == 'CL':
			cl_dirs.append(config[section][option])

os.chdir(PROJ_PATH)
print(os.getcwd())
dir_prefix = os.getcwd() + '/' 

for dirs in bs_dirs:
	dirs = dirs.replace('../../', dir_prefix)
	print(dirs)
	try:
		os.makedirs(dirs, mode=0o764, exist_ok=False)
	except OSError as err:
		print(f"Custom Error: directory {dirs} already exists")

for dirs in ls_dirs:
	print(dirs)
	dirs = dirs.replace('../../', dir_prefix)
	try:
		os.makedirs(dirs, mode=0o764, exist_ok=False)
	except OSError as err:
		print(f"Custom Error: directory {dirs} already exists")

for dirs in cl_dirs:
	print(dirs)
	dirs = dirs.replace('../../', dir_prefix)
	try:
		os.makedirs(dirs, mode=0o764, exist_ok=False)
	except OSError as err:
		print(f"Custom Error: directory {dirs} already exists")
"""