from configparser import ConfigParser
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os

parser = ConfigParser()
parser.read("../../Other/ConfigParser/config.ini")
IMGAE_HEIGHT, IMAGE_WIDTH = int(parser.get('BS','image_height')),int(parser.get('BS','image_width'))

class BoneSuppressionDataset(Dataset):
	"""
		Xrays and Masks loader.
		Masks : BSE_Xrays
	"""
	def __init__(self, Xray_dir, mask_dir, transform=None, test=False):
		"""
			### Inputs ###
			Xray_dir : directory with original Xray images
			mask_dir : directory with BSE Xray images
		"""
		self.Xray_dir = Xray_dir
		self.mask_dir = mask_dir
		self.transform = transform
		self.Xrays = os.listdir(Xray_dir)
		self.test = test

	def __len__(self):
		"""
			Returns total number of Xrays
		"""
		return len(self.Xrays)

	def __getitem__(self,index):
		"""
			### Input ###
			index : index of the Xray from the dataset

			### Returns ###
			xray : original Xray image
			mask : BSE Xray image
		"""
		xray_path = os.path.join(self.Xray_dir, self.Xrays[index])
		mask_path = os.path.join(self.mask_dir, self.Xrays[index])
		xray = np.asarray(Image.open(xray_path), dtype=np.float32)
		xray = np.expand_dims(xray,-1)
		mask = np.asarray(Image.open(mask_path), dtype=np.float32) # 0-255.0
		mask = np.expand_dims(mask,-1)
		if self.test:
			return xray, mask

		if self.transform is not None:
			augmentations = self.transform(image=xray, mask=mask)
			xray = augmentations['image']
			mask = augmentations['mask']
		return xray, mask

def main():
	dataset = BoneSuppressionDataset("../../Data/BS/train/Xrays/", "../../Data/BS/train/BSE_Xrays/", test=True)
	print("="*50)
	print("Bone Suppression Data generator test : ")
	print("*"*20)
	print("Number of Xrays : ",len(dataset))
	print("Shape of original Xray : ",dataset[0][0].shape)
	print("Shape of BSE Xray : ",dataset[0][1].shape)
	print("="*50)
	assert dataset[0][0].shape == dataset[0][1].shape, "Data-Target shapes not equal."

if __name__=="__main__":
	main()
