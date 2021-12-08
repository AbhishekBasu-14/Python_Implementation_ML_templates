# Convolutional Neural Network

# Importing the libraries
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
tf.__version__

# Part 1 - Data Preprocessing

# Preprocessing the Training set
# Feature Scaling is necessary for CNN.
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
# Geometric transformation are applied as shown on keras website code template.  
# rescale applies feature scaling.
                                   
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

# Preprocessing the Test set
test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

# Part 2 - Building the CNN

# Initialising the CNN
cnn = tf.keras.models.Sequential()

# Step 1 - Convolution
# For images, its like we are finding feature map matrices to perform hitting operation.
# The layer has got number of matrices for filtering operation.
# ReLU helps in gathering linear relationships.
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))

# Step 2 - Pooling
# Stride specifies the size (length, width) of filter.
# By max pooling, we reduce matrix size by preserve only the maximum feature values. 
# The helps the model to focus on most relevant features for identification.
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Adding a second convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Step 3 - Flattening
# For arranging the neural nets matrix in a column.
cnn.add(tf.keras.layers.Flatten())
# After flattening, we proceed with implementing ANN to our column vectors.

# Step 4 - Full Connection
# Starting to apply ANN
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
# Dense class helps in creating a fully connected neural layer
# Units: number of neurons (by trial and error)

# Step 5 - Output Layer
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
# Sigmoid for binary classification
# Softmax for non-binary classifications

# Part 3 - Training the CNN

# Compiling the CNN
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# Adam is one of the best optimizer, helps in updating weights and reducing losses.
# For binary outcomes- use binary_crossentropy.
# In case we had non-binary, use categorical_crossentropy
# The logarithmic function in cross entropy helps in backpropogation and gradient descent.

# Training the CNN on the Training set and evaluating it on the Test set
cnn.fit(x = training_set, validation_data = test_set, epochs = 25)

# Part 4 - Making a single prediction

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
print(prediction)
