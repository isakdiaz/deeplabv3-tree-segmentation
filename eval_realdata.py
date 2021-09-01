
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import cv2
import pandas as pd
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
from metrics import dice_loss, dice_coef, iou


""" Global parameters """
H = 512
W = 512

""" Creating a directory """
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_data(path):
    x = sorted(glob(os.path.join(path, "*jpg")))
    return x


def save_results(image, y_pred, save_image_path):
    ## i - m - yp - yp*i
    line = np.ones((H, 10, 3)) * 128

    y_pred = np.expand_dims(y_pred, axis=-1)    ## (512, 512, 1)
    y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=-1)  ## (512, 512, 3)

    masked_image = image * y_pred
    y_pred = y_pred * 255

    cat_images = np.concatenate([image,  line, y_pred, line, masked_image], axis=1)
    cv2.imwrite(save_image_path, cat_images)

def resize_data(image, size=(512,512)):
    H = size[0]
    W = size[1]

    
    x_min, y_min = 0, 0
    print(image.shape)
    if image.shape[0] < image.shape[1]:
        x_min = (image.shape[1] - image.shape[0]) // 2
    elif image.shape[0] > image.shape[1]:
        y_min = (image.shape[0] - image.shape[1]) // 2


    min_dim = min(image.shape[0], image.shape[1])

    image = image[y_min:y_min + min_dim, x_min:x_min+min_dim]

    print(image.shape)
    image = cv2.resize(image, (W, H))
    print(image.shape)
    print("_----------------------")
    return image

def change_aspect_ratio(image):
    """
    input: image as numpy array
    returns: image with 1:1 aspect ratioo
    """
    x_min, y_min = 0, 0
    print(image.shape)
    if image.shape[0] < image.shape[1]:
        x_min = (image.shape[1] - image.shape[0]) // 2
    elif image.shape[0] > image.shape[1]:
        y_min = (image.shape[0] - image.shape[1]) // 2


    min_dim = min(image.shape[0], image.shape[1])

    return image[y_min:y_min + min_dim, x_min:x_min+min_dim]



if __name__ == "__main__":

    # MODEL_FILENAME = "files/treeseg2021-08-29.h5"
    GDRIVE_DIR = "/content/drive/MyDrive/saved_models/treeseg" # Save Path for models
    # MODEL_NAME = "treeseg2021-08-29.h5"
    MODEL_NAME = "treeseg_2021-09-01.h5"
    MODEL_FILENAME = os.path.join(GDRIVE_DIR, MODEL_NAME)
    
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Directory for storing files """
    create_dir("results_realdata")

    """ Loading model """
    with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss}):
        model = tf.keras.models.load_model(MODEL_FILENAME)

    """ Load the dataset """
    dataset_path = "data"
    valid_path = os.path.join(dataset_path, "test")
    test_x = load_data(valid_path)
    print(f"Test: {len(test_x)}")

    """ Evaluation and Prediction """
    for x in tqdm(test_x, total=len(test_x)):
        """ Extract the name """
        name = x.split("/")[-1].split(".")[0]

        """ Reading the image """
        image = cv2.imread(x, cv2.IMREAD_COLOR)
        image = resize_data(image)
        x = image/255.0
        x = np.expand_dims(x, axis=0)
        """ Prediction """
        y_pred = model.predict(x)[0]
        y_pred = np.squeeze(y_pred, axis=-1)
        y_pred = y_pred > 0.5
        y_pred = y_pred.astype(np.int32)

        """ Saving the prediction """
        save_image_path = f"results_realdata/{name}.png"
        save_results(image, y_pred, save_image_path)



