#Image classifier using Convolutional Neural Networks (CNN) in Keras and TensorFlow
#Import Necessary Libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

#Load MNIST dataset
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

#Preprocess: Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

#Reshape images to (28, 28, 1) as they are grayscale
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

#Convert labels to one-hot encoded format
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

#Build CNN model
model = models.Sequential()

#First Conv layer
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))

#Second Conv layer
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

#Third Conv layer
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

#Flatten the 3D output to 1D and add Dense layers
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))

#Output layer with 10 neurons(for 10 digit classes)
model.add(layers.Dense(10, activation='softmax'))

#Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#Train the model
model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_data=(test_images, test_labels))

#Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc * 100:.2f}%")

#Make predictions
predictions = model.predict(test_images)
print(f"Prediction for first test image: {np.argmax(predictions[0])}")

plt.imshow(test_images[0].reshape(28, 28), cmap='gray')
plt.title(f"Predicted Label: {predictions[0].argmax()}")
plt.show()
