
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.callbacks import *
from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow.keras.metrics import *
from glob import glob
from sklearn.model_selection import train_test_split
from model import build_model
from doubleunet_pytorch import build_doubleunet
from utils import *
from metrics import *

def read_image(x):
    x = x.decode()
    image = cv2.imread(x, cv2.IMREAD_COLOR)
    # image = cv2.resize(image, (512, 384))  # Resize to the expected shape
    # print(f"Original image shape: {image.shape}")
    image = np.clip(image - np.median(image)+127, 0, 255)
    image = image/255.0
    image = image.astype(np.float32)
    # print(f"Processed image shape: {image.shape}")
    return image

def read_mask(y):
    y = y.decode()
    mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
    # mask = cv2.resize(mask, (512, 384))  # Resize to the expected shape
    # print(f"Original mask shape: {mask.shape}")
    mask = mask/255.0
    mask = mask.astype(np.float32)
    mask = np.expand_dims(mask, axis=-1)
    # print(f"Processed mask shape: {mask.shape}")
    return mask

def parse_data(x, y):
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        y = np.concatenate([y, y], axis=-1)
        return x, y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
    # x.set_shape([384, 512, 3])
    # y.set_shape([384, 512, 2])
    x.set_shape([192, 256, 3])
    y.set_shape([192, 256, 2])
    return x, y

def tf_dataset(x, y, batch=8):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.shuffle(buffer_size=32)
    dataset = dataset.map(map_func=parse_data)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch)
    # dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

def print_dataset_shapes(dataset):
    for images, masks in dataset.take(1):  # Take one batch from the dataset
        print(f"Images shape: {images.shape}")
        print(f"Masks shape: {masks.shape}")

if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)
    create_dir("files")

    # train_path = "new_data/train/"
    # valid_path = "new_data/valid/"
    train_path = "new_data/train"
    valid_path = "new_data/valid"

    ## Training
    train_x = sorted(glob(os.path.join(train_path, "image", "*.jpg")))
    train_y = sorted(glob(os.path.join(train_path, "mask", "*.jpg")))

    print(len(train_x))
    print(len(train_y))
    ## Shuffling
    # train_x, train_y = shuffling(train_x, train_y)

    ## Validation
    valid_x = sorted(glob(os.path.join(valid_path, "image", "*.jpg")))
    valid_y = sorted(glob(os.path.join(valid_path, "mask", "*.jpg")))

    model_path = "files/model.keras"
    batch_size = 16
    # epochs = 300
    epochs = 30
    lr = 1e-4
    # shape = (384, 512, 3)
    shape = (192, 256, 3)

    # tf.keras.mixed_precision.set_global_policy('mixed_float16')

    print("Before build_model")
    model = build_model(shape)
    metrics = [
        dice_coef,
        iou,
        Recall(),
        Precision()
    ]
    # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    #     initial_learning_rate=lr,
    #     decay_steps=100,
    #     decay_rate=0.96,
    #     staircase=True)
    print("After build_model")
    train_dataset = tf_dataset(train_x, train_y, batch=batch_size)
    valid_dataset = tf_dataset(valid_x, valid_y, batch=batch_size)
    print("Before model.compile")
    # model.compile(loss=dice_loss, optimizer=Adam(lr), metrics=metrics)
    model.compile(loss=dice_loss, optimizer=Adam(lr), metrics=['accuracy'])
    
    print("Before callbacks")
    callbacks = [
        ModelCheckpoint(model_path),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=20),
        CSVLogger("files/data.csv"),
        TensorBoard(),
        EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=False)
    ]
    print("After callbacks")
    train_steps = (len(train_x)//batch_size)
    valid_steps = (len(valid_x)//batch_size)

    if len(train_x) % batch_size != 0:
        train_steps += 1

    if len(valid_x) % batch_size != 0:
        valid_steps += 1
    print("train_steps: ", train_steps)
    print("valid_steps: ", valid_steps)
    # print_dataset_shapes(train_dataset)
    # print_dataset_shapes(valid_dataset)
    # model.summary()
    # model.fit(train_dataset, epochs=2, steps_per_epoch=10)
    model.fit(train_dataset,
            epochs=epochs,
            validation_data=valid_dataset,
            steps_per_epoch=train_steps,
            validation_steps=valid_steps,
            callbacks=callbacks,
            shuffle=False)
