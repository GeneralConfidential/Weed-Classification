from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np
import os

train_dir = "C:/Users/raagg/OneDrive/Documents/GitHub/Weed-Classification/data"

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

batch_size = 32
target_size = (224, 224)

for directory in [train_dir]:
    generator = datagen.flow_from_directory(
        directory,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)
    num_images = len(generator.filenames)
    num_batches = int(num_images / batch_size) + 1
    for i in range(num_batches):
        batch = next(generator)
        images, labels = batch
        for j in range(len(images)):
            image = cv2.cvtColor(images[j], cv2.COLOR_BGR2RGB)
            label = labels[j]
            label_name = list(generator.class_indices.keys())[np.argmax(label)]
            original_path = os.path.join(directory, label_name)
            new_path = os.path.join(directory , label_name)
            if not os.path.exists(new_path):
                os.makedirs(new_path)
            image_path = os.path.join(new_path,'augmented{}.jpeg'.format(i*batch_size+j+1))
            cv2.imwrite(image_path, image)
