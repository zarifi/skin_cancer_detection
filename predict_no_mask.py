
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import CustomObjectScope
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from utils import *
from train import tf_dataset

def read_image(x):
    image = cv2.imread(x, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (256, 192))
    image = np.clip(image - np.median(image)+127, 0, 255)
    image = image/255.0
    image = image.astype(np.float32)
    image = np.expand_dims(image, axis=0)
    return image

def read_mask(y):
    mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
    mask = mask.astype(np.float32)
    mask = mask/255.0
    mask = np.expand_dims(mask, axis=-1)
    return mask

def mask_to_3d(mask):
    mask = np.squeeze(mask)
    mask = np.stack((mask,)*3, axis=-1)
    # mask = [mask, mask, mask]
    # mask = np.transpose(mask, (1, 2, 0))
    return mask

def parse(y_pred):
    y_pred = np.expand_dims(y_pred, axis=-1)
    y_pred = y_pred[..., -1]
    y_pred = y_pred.astype(np.float32)
    y_pred = np.expand_dims(y_pred, axis=-1)
    return y_pred

def evaluate_normal(model, x_data, dataset):
    THRESHOLD = 0.5
    total = []
    create_dir(f"results_predictions/PAD/{dataset}/image")
    create_dir(f"results_predictions/PAD/{dataset}/mask")
    create_dir(f"results_predictions/PAD/{dataset}/cropped")
    for i, x in tqdm(enumerate(x_data), total=len(x_data)):
        x_id = x.split("\\")[-1].replace(".png", "")
        x = read_image(x)
        _, h, w, _ = x.shape
        y_pred = parse(model.predict(x)[0][..., -1])
        image = x[0] * 255.0
        mask = mask_to_3d(y_pred) * 255.0

        # cropped_image = np.array(image) * np.array(mask)
        cropped_image = cv2.bitwise_and(image, mask)
        
        cv2.imwrite(f"results_predictions/PAD/{dataset}/image/{x_id}.png", image)
        cv2.imwrite(f"results_predictions/PAD/{dataset}/mask/{x_id}.png", mask)
        cv2.imwrite(f"results_predictions/PAD/{dataset}/cropped/{x_id}.png", cropped_image)

smooth = 1.
def dice_coef(y_true, y_pred):
    y_true_f = tf.keras.layers.Flatten()(y_true)
    y_pred_f = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)
    

    batch_size = 8
    path = "..\\PAD-UFES-20_NEW\\images\\"
    train_x = sorted(glob(os.path.join(path, "imgs_part_1", "*.png")))
    # test_x = sorted(glob(os.path.join(path, "ISIC2018_Task3_Test_Input", "*.jpg")))
    # train_steps = (len(train_x)//batch_size)
    # if len(train_x) % batch_size != 0:
    #     train_steps += 1
    print(len(train_x))
    model = load_model_weight("files/model.keras")
    # model.evaluate(test_x, steps=test_steps)
    # print(test_x[0].split("\\")[-1].replace(".jpg", ""))
    evaluate_normal(model, train_x[:100], "train")
    # evaluate_normal(model, test_x, "test")
