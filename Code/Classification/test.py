from configparser import ConfigParser
import pandas as pd

parser = ConfigParser()
parser.read("../Other/ConfigParser/config.ini")

csv_path = parser.get('CL','train_csv_pathology')

df = pd.read_csv(csv_path)
print(df.head())