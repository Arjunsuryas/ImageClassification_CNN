import numpy as np
from tensorflow.keras import datasets  

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()


x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0


x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)


print("Training data shape:", x_train.shape)  
print("Test data shape:", x_test.shape)       
