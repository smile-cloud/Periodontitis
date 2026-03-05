import os

image_list = os.listdir("image")
with open("image.txt", "w") as f:
    for image in image_list:
        image = image.split(".nii")[0]
        f.write(image + "\n")