import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#load the data set
#loads dataset from tensorflow

mnist = tf.keras.datasets.mnist
# #get the labeled data so that we know with all the digits
# #labeled data means that we already know, and what the digits are without guessing them 
# # since its a large amount of data, the program splits up the data and trains the model using that training data
# #testing data asseses the model and to see how well the model performs on data that it has never seen before

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# # #normalizing which means the model will scale the data down 
# # #so that there is a value between 0  and 1 in rgb values since black and white is only 0-1

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis =1)

model = tf.keras.models.Sequential([
    # Reshape input to be a 4D tensor.
    tf.keras.layers.Reshape((28, 28, 1), input_shape=(28, 28)),
    # First convolutional layer with pooling.
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    # Second convolutional layer with pooling.
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    # Flatten the tensor to feed into a dense layer.
    tf.keras.layers.Flatten(),
    # Dense layer with relu activation.
    tf.keras.layers.Dense(64, activation='relu'),
    # Output layer with softmax activation.
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model with optimizer, loss function and metric.
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Fit the model to the training data.
model.fit(x_train, y_train, epochs=5)

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-2,
    decay_steps=10000,
    decay_rate=0.9)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)


model.add(tf.keras.layers.BatchNormalization())



model.save('handwritten.keras')  


model = tf.keras.models.load_model('handwritten.keras')





image_number =1

while os.path.isfile(f"Digits/digit{image_number}.png"):
    try:
        # Read the image in grayscale
        img = cv2.imread(f"Digits/digit{image_number}.png")[:,:,0]
        # Invert the image so that the digit is black and the background is white
        img = np.invert(np.array([img]))
        # Normalize the inverted image
        img_normalized = img / 255.0
        # Reshape the image to fit the model input, add a batch dimension and a channel dimension if needed
        img_reshaped = np.expand_dims(img_normalized, axis=[0, -1])
        # Make a prediction
        prediction = model.predict(img)
        print(f"the number is probably: {np.argmax(prediction)}")
        # Display the original inverted image for visualization
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        image_number += 1




