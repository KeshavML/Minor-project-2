from configparser import ConfigParser
import os

parser = ConfigParser()
config_file_path = './Other/ConfigParser/config.ini'
parser.read(config_file_path)


# SET 1
parser.get('LS','save_model_path').replace("../../","./")
parser.get('LS','save_images').replace("../../","./")
parser.get('LS','train_img_dir').replace("../../","./")
parser.get('LS','train_mask_dir').replace("../../","./")
parser.get('LS','val_img_dir').replace("../../","./")
parser.get('LS','val_mask_dir').replace("../../","./")

parser.get('LS','save_images_pred_path').replace("../../","./")
parser.get('LS','load_images_pred_path').replace("../../","./")


# SET 2
parser.get('LS','').replace('../../','./load_model_path')
parser.get('LS','').replace('../../','./save_images')
parser.get('LS','').replace('../../','./train_img_dir')
parser.get('LS','').replace('../../','./train_mask_dir')
parser.get('LS','').replace('../../','./val_img_dir')
parser.get('LS','').replace('../../','./val_mask_dir')

parser.get('LS','').replace('../../','./save_images_pred_path')
parser.get('LS','').replace('../../','./load_images_pred_path')
parser.get('LS','').replace('../../','./save_images_pred_covid')
parser.get('LS','').replace('../../','./load_images_pred_covid')

# SET 3
parser.get('CL','').replace('../../','./save_model_path_pathology')
parser.get('CL','').replace('../../','./save_images_pathology')
parser.get('CL','').replace('../../','./train_img_dir_pathology')
parser.get('CL','').replace('../../','./train_csv_pathology').replace("train_pathology.csv",'')
parser.get('CL','').replace('../../','./val_img_dir_pathology')
parser.get('CL','').replace('../../','./val_csv_pathology').replace("val_pathology.csv",'')
parser.get('CL','').replace('../../','./pred_csv_pathology')

parser.get('CL','').replace('../../','./save_model_path_covid')
parser.get('CL','').replace('../../','./save_images_covid')
parser.get('CL','').replace('../../','./train_img_dir_covid')
parser.get('CL','').replace('../../','./train_csv_covid').replace("train_covid.csv",'')
parser.get('CL','').replace('../../','./val_img_dir_covid')
parser.get('CL','').replace('../../','./val_csv_covid').replace("val_covid.csv",'')
parser.get('CL','').replace('../../','./pred_csv_covid')

parser.get('CL','').replace('../../','./dataset_path')
parser.get('CL','').replace('../../','./dataset_covid')
parser.get('CL','').replace('../../','./train_img_dir_bs_path')
parser.get('CL','').replace('../../','./train_img_dir_bs_covid')
parser.get('CL','').replace('../../','./train_img_dir_ls_path')
parser.get('CL','').replace('../../','./train_img_dir_ls_covid')







