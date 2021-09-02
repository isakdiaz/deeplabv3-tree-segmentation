"""
Renders and backgrounds must be merged  to form images.
Corresponding masks must also be cropped to correct dimensions.

"""


import os
from albumentations.augmentations.crops.functional import crop
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import random
import skimage


""" Creating a directory """
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_data(path):
    
    # Get names in path directory
    bg_names = sorted(glob(os.path.join(path, "background", "*.jp*g")))
    fg_names = sorted(glob(os.path.join(path, "foreground", "*.png")))
    mask_names = sorted(glob(os.path.join(path, "rendered_mask", "*.png")))

    # # Load images into array
    bgs = [cv2.imread(name, cv2.IMREAD_COLOR) for name in bg_names]
    fgs = [cv2.imread(name, cv2.IMREAD_UNCHANGED) for name in fg_names]
    masks = [cv2.imread(name, cv2.IMREAD_GRAYSCALE) for name in mask_names]

    # Convert masks to 0 or 1
    mask_res = []
    for mask in masks:
        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        mask_res.append(mask)

    return (bgs, fgs, mask_res)


def center_crop(img, dim):
	"""Returns center cropped image
	Args:
	img: image to be center cropped
	dim: dimensions (width, height) to be cropped
	"""
	height, width = img.shape[0], img.shape[1]

	# process crop width and height for max available dimension
	crop_width = dim[1] if dim[1]<img.shape[1] else img.shape[1]
	crop_height = dim[0] if dim[0]<img.shape[0] else img.shape[0] 
	mid_x, mid_y = int(width/2), int(height/2)
	cw2, ch2 = int(crop_width/2), int(crop_height/2) 
	crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
	return crop_img

def scale_image(img, factor=1):
	"""Returns resize image by scale factor.
	This helps to retain resolution ratio while resizing.
	Args:
	img: image to be scaled
	factor: scale factor to resize
	"""
	return cv2.resize(img,(int(img.shape[1]*factor), int(img.shape[0]*factor)))


def resize_images(imgs, size=(720, 720)):
    """
    Returns array with all images of same size.
    Upscales images if necessary before performing center crop.
    """
    res = []
    for img in imgs:

        factor = max(1, size[0]/float(img.shape[0]), size[1]/float(img.shape[1]))
        if factor != 1: img = scale_image(img, factor)

        img = center_crop(img, size)
        res.append(img)

    return res

def left_crop(images, width=720):
    for i in range(len(images)):
        images[i] = images[i][:,:width]
        # print(images[i].shape)
    return images

def merge(bgs, fgs):

    """
    Merge two pictures of the same size,
    Backgrounds are chosen at random.
    """
    # random.seed(42)

    bgs_len = len(bgs)

    res = []
    for fg_index in tqdm(range(len(fgs))):
        fg = fgs[fg_index]
        bg = bgs[random.randint(0, bgs_len - 1)]
        new_img = bg.copy()
        for i in range(new_img.shape[0]):
            for j in range(new_img.shape[1]):
                if fg[i][j][-1] > 0:
                    new_img[i][j] = fg[i][j][:3]
        
        res.append(new_img)
        
    return res
        


def save_images(imgs, path, prefix, file_type="jpg", start_num=0):

    for img in imgs:
        # print(img.shape)
        cv2.imwrite(os.path.join(path, f"{prefix}_{str(start_num).zfill(5)}.{file_type}"), img)
        start_num += 1

def save_test_image(img, prefix="test", path= "trash", file_type="jpg"):
    random_ind = np.random.randint(0,100)
    cv2.imwrite(os.path.join(path, f"{prefix}_{random_ind}.{file_type}"), img)

def salt_and_pepper_noise(image, prob):
    """
    Add salt and pepper noise to image
    prob: Probability of the noise
    """
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output

def random_noise(image, prob = 0.75):
    """
    input: rgb image 0-255, probability of adding noise
    output: rgb image 0-255

    """

    if(np.random.random() > prob): return image
    modes = ["gaussian", "poisson", "speckle", "salt", "pepper", "s&p"]
    ind = np.random.randint(0, len(modes))

    # Salt and pepper noise requires extra parameter that crashes other modes
    if modes[ind] in ["salt", "pepper", "s&p"]:
        image = skimage.util.random_noise(image, mode=modes[ind], amount=0.005)
    else:
        image = skimage.util.random_noise(image, mode=modes[ind])

    # Skimage converts image from 0-255 to 0-1.0, need to convert back
    # If you don't return array to int, RAM usage will explode
    image = (image * 255).astype(np.uint8, casting='unsafe')

    return image

def add_noise(images, prob=0.75):

    for i in range(len(images)):
        images[i] = random_noise(images[i], prob=prob)
    
    return images

if __name__ == "__main__":
    
    MASKS_DIR = "data/mask"
    IMAGES_DIR = "data/image"
    PREFIX = "oaktree"

    print("Creating Directories...")
    create_dir("trash") # folder for troubleshooting
    create_dir(MASKS_DIR)
    create_dir(IMAGES_DIR)


    print("Loading Images...")
    bgs, fgs, masks = load_data("data")

    print("Cropping Images to Squares...")
    fgs = left_crop(fgs, 720)
    masks = left_crop(masks, 720)

    
    print("Adding noise to foregrounds...")
    fgs = add_noise(fgs)

    print("Adding noise to backgrounds...")
    bgs = add_noise(bgs)

    # print("Saving fgs...")
    # save_images(fgs, "trash", "test", file_type="png")
 
    print("Saving Masks...")
    save_images(masks, MASKS_DIR, PREFIX, file_type="png")

    print("Resizing Images...")
    bgs = resize_images(bgs)

    print("Merging foregrounds and backgrounds...")
    imgs = merge(bgs, fgs)

    print("Saving Images...")
    save_images(imgs, IMAGES_DIR, PREFIX, file_type="jpg")