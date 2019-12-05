import os

os.chdir("/Users/PycharmProjects/Tinder_Chad/")
image_path = "cropped_largest_resized"

path = f"Analysis/{image_path}/"
for old in os.listdir(path):
    try:
        new = old.split('_')[0]
        os.rename(f"{path}/{old}", f"{path}/{new}.jpg")
    except ValueError:
        pass
