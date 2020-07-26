import os
from draw import check_images
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_horse_dir = './data/horse-or-human/horses'
train_human_dir = './data/horse-or-human/humans'

train_horse_names = os.listdir(train_horse_dir)
train_human_names = os.listdir(train_human_dir)

# check_images(4, 4, [train_horse_dir, train_human_dir],
#  [train_horse_names, train_human_names])

model = tf.keras.models.Sequential([
    layers.Conv2D(16, (3, 3), activation='relu', input_shape=(300, 300, 3)),
    layers.MaxPool2D(2, 2),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPool2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPool2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPool2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPool2D(2, 2),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.summary()

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory(
    './data/horse-or-human',
    target_size=(300, 300),
    batch_size=128,
    class_mode='binary'
)

history = model.fit(
    train_generator,
    steps_per_epoch=8,
    epochs=15,
    verbose=1
)
