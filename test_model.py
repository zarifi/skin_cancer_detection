import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.optimizers import Adam
from model import build_model
from tensorflow.keras.metrics import *
from utils import *
from metrics import *

# Define your model functions here (copy the entire model definition code from the previous response)

def preprocess_data(data):
    """ Preprocess the dataset to be used in model training """
    image = tf.image.resize(data['image'], (192, 256))
    mask = tf.image.resize(data['segmentation_mask'], (192, 256))

    image = tf.cast(image, tf.float32) / 255.0
    mask = tf.cast(mask, tf.float32) / 255.0

    return image, mask

def build_and_compile_model():
    """ Build and compile the model """
    model = build_model((192, 256, 3))
    # model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    metrics = [
        dice_coef,
        iou,
        Recall(),
        Precision()
    ]
    model.compile(loss=dice_loss, optimizer=Adam(1e-4), metrics=['accuracy'])
    return model

def main():
    # Load the Oxford-IIIT Pet Dataset
    dataset, info = tfds.load('oxford_iiit_pet:3.2.0', with_info=True)
    train_dataset = dataset['train'].map(preprocess_data, num_parallel_calls=tf.data.AUTOTUNE).batch(8).prefetch(tf.data.AUTOTUNE)
    test_dataset = dataset['test'].map(preprocess_data, num_parallel_calls=tf.data.AUTOTUNE).batch(8).prefetch(tf.data.AUTOTUNE)
    print(train_dataset)
    # Build and compile the model
    model = build_and_compile_model()
    
    # Train the model
    model.fit(train_dataset, validation_data=test_dataset, epochs=10)

    # Save the model
    # model.save('segmentation_model.h5')

if __name__ == "__main__":
    main()
