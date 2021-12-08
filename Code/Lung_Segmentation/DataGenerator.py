from configparser import ConfigParser
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os

parser = ConfigParser()
parser.read("../../Other/ConfigParser/config.ini")
IMGAE_HEIGHT, IMAGE_WIDTH = int(parser.get('BS','image_height')),int(parser.get('BS','image_width'))

class LungSegmentationDataset(Dataset):
	"""
		Xrays and Masks loader.
	"""
	def __init__(self, Xray_dir, mask_dir, transform=None, test=False):
		"""
			### Input ###
			Xray_dir : directory with original xrays
			mask_dir : directory with masks
		"""
		self.Xray_dir = Xray_dir
		self.mask_dir = mask_dir
		self.transform = transform
		self.Xrays = os.listdir(Xray_dir)
		self.test = test

	def __len__(self):
		"""
			Returns : number of xrays.
		"""
		return len(self.Xrays)

	def __getitem__(self,index):
		"""
			### Input ###
			index : index from the xrays list

			### Return ###
			xray : Xray image
			mask : mask for the xray
		"""
		xray_path = os.path.join(self.Xray_dir, self.Xrays[index])
		mask_path = os.path.join(self.mask_dir, self.Xrays[index])
		xray = np.array(Image.open(xray_path), dtype=np.float32)
		xray = np.expand_dims(xray,-1)
		mask = np.array(Image.open(mask_path), dtype=np.float32) # 0-255.0
		mask = np.expand_dims(mask,-1)
		# mask[mask >= 10.0] = 1.0

		if self.test:
			return xray, mask

		if self.transform is not None:
			augmentations = self.transform(image=xray, mask=mask)
			xray = augmentations['image']
			mask = augmentations['mask']
		return xray, mask

def main():
	dataset = LungSegmentationDataset("../../Data/LS/train/Xrays/", "../../Data/LS/train/Masks/", test=True)
	print("="*50)
	print("Lung Segmentation Data generator test : ")
	print("*"*20)
	print("Number of Xrays : ",len(dataset))
	print("Shape of original Xray : ",dataset[0][0].shape)
	print("Shape of Mask : ",dataset[0][1].shape)
	print("="*50)
	print("Xray data: ",type(dataset[0][1]))
	print("="*50)
	# from matplotlib import pyplot as plt
	# plt.imshow(dataset[0][1], interpolation='nearest')
	# plt.imshow(dataset[0][0], interpolation='nearest')
	# plt.show()
	assert dataset[0][0].shape == dataset[0][1].shape, "Data-Target shapes not equal."

if __name__=="__main__":
	main()
