import numpy as np
import random
import cv2
import albumentations as A
import glob

images_to_generate = 1000

images_path = "data/raw/images/images/"  # path to original images
masks_path = "data/raw/images/groundtruth/"
img_augmented_path = "data/interim/augmented_data/images/"  # path to store aumented images
msk_augmented_path = "data/interim/augmented_data/groundtruth/"  # path to store aumented images
images = []  # to store paths of images from folder
masks = []

filenames = glob.glob(images_path + '*.png')
filenames.sort()

for im in filenames:  # read image name from folder and append its path into "images" array
    images.append(im)

filenames = glob.glob(masks_path + '*.png')
filenames.sort()
for msk in filenames:  # read image name from folder and append its path into "images" array
    masks.append(msk)

auxlistim = [i.split('/')[1] for i in images]
auxlistmask = [i.split('/')[1] for i in masks]
excess = list(set(auxlistim).difference(auxlistmask))

for name in excess:
    images.remove(images_path + name)

aug = A.Compose([
    A.VerticalFlip(p=0.5),

    # A.RandomRotate90(p=0.5),
    # A.HorizontalFlip(p=1),
    # A.Transpose(p=1),
   # A.ElasticTransform(p=0.2, alpha=200, sigma=200 * 0.08, alpha_affine=60 * 0.09),
    # A.GridDistortion(p=1),
]
)

i = 1  # variable to iterate till images_to_generate

while i <= images_to_generate:
    number = random.randint(0, len(images) - 1)  # PIck a number to select an image & mask
    image = images[number]
    mask = masks[number]

    # image=random.choice(images) #Randomly select an image name
    original_image = np.array(cv2.imread(image))
    original_mask = np.array(cv2.imread(mask))

    augmented = aug(image=original_image, mask=original_mask)
    transformed_image = augmented['image']
    transformed_mask = augmented['mask']

    new_image_path = "%s/augmented_image_%s.png" % (img_augmented_path, i)
    new_mask_path = "%s/augmented_mask_%s.png" % (msk_augmented_path, i)
    cv2.imwrite(new_image_path, cv2.cvtColor(transformed_image, cv2.COLOR_BGR2GRAY))
    cv2.imwrite(new_mask_path, cv2.cvtColor(transformed_mask, cv2.COLOR_BGR2GRAY))
    i = i + 1
