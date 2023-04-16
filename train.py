import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from sklearn.model_selection import train_test_split
import os
import numpy as np

# Define constants
num_classes = 3
input_shape = (224, 224, 3)
batch_size = 32
epochs = 100

# Load images
image_dir = 'data/'
X = []
y = []
for i, weed in enumerate(os.listdir(image_dir)):
    for image_file in os.listdir(os.path.join(image_dir, weed)):
        image = tf.keras.preprocessing.image.load_img(
            os.path.join(image_dir, weed, image_file), 
            target_size=input_shape[:2]
        )
        image = tf.keras.preprocessing.image.img_to_array(image)
        X.append(image)
        y.append(i)
X = np.array(X)
y = tf.keras.utils.to_categorical(y, num_classes)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    vertical_flip=True)

val_datagen = ImageDataGenerator(rescale=1./255)

# Load images
train_generator = train_datagen.flow(
    X_train,
    y_train,
    batch_size=batch_size,
    shuffle=True)

val_generator = val_datagen.flow(
    X_test,
    y_test,
    batch_size=batch_size,
    shuffle=True)

# Define the model architecture
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(train_generator, epochs=epochs, validation_data=val_generator)
model.save('model.h5')

#Done by Raag Gautam 20BCE7144