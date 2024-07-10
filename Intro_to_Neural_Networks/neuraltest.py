import tensorflow as tf
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense

from tensorflow.keras.callbacks import EarlyStopping


model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()


# Load the dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(
    train_images, train_labels,
    epochs=50,  # Use a larger number of epochs
    validation_split=0.2,  # Use 20% of the data for validation
    callbacks=[early_stopping]
)


# Display the shape of the dataset
print(f'Train images shape: {train_images.shape}')
print(f'Test images shape: {test_images.shape}')

# Normalize the images to a range of 0 to 1
train_images = train_images / 255.0
test_images = test_images / 255.0



model.fit(train_images, train_labels, epochs=5)
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_acc}')

predictions = model.predict(test_images)

# Display the first test image and the model's prediction
plt.imshow(test_images[0], cmap='gray')
#plt.title(f'Predicted Label: {tf.argmax(predictions[0])}, True Label: {test_labels[0]}')
# Display the first image and label
plt.imshow(train_images[0], cmap='gray')
plt.title(f'Label: {train_labels[0]}')
plt.show()
