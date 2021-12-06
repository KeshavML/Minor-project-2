import os

masks = os.listdir("./Masks")
masks = [f"./Masks/{each}" for each in masks] 
for each in masks:
    filename, fileext = os.path.splitext(each)
    # print(filename,fileext)
    filename = filename.replace("_mask","") + fileext
    os.rename(each, filename)
    # print(filename)
