import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

image_files = [
    'images.jpg',
    'aic-home (1).jpg',
    'jpegls-home (1).jpg',
    'Un_super_paysage.jpg',
    'what-is-jpeg-format (1).jpg'
]

kernel = tf.constant([
    [-1., -1., -1.],
    [-1.,  8., -1.],
    [-1., -1., -1.]
], dtype=tf.float32)
kernel = tf.reshape(kernel, [3, 3, 1, 1])

processed_images = []
valid_image_files = []

for image_path in image_files:
    if os.path.exists(image_path):
        # Read and preprocess
        image = tf.io.read_file(image_path)
        image = tf.io.decode_jpeg(image, channels=1)
        image = tf.image.resize(image, [28, 28])
        image = tf.image.convert_image_dtype(image, tf.float32)
        processed_images.append(image)
        valid_image_files.append(image_path)
        
        plt.figure(figsize=(3,3))
        plt.imshow(image.numpy().squeeze(), cmap='gray')
        plt.axis('off')
        plt.title(f'Original: {image_path}')
        plt.show()
        
        image_batch = tf.expand_dims(image, axis=0)  # add batch dimension
        conv = tf.nn.conv2d(image_batch, filters=kernel, strides=1, padding='SAME')
        plt.figure(figsize=(3,3))
        plt.imshow(tf.squeeze(conv), cmap='gray')
        plt.axis('off')
        plt.title('Edge Detection')
        plt.show()
        
        relu = tf.nn.relu(conv)
        plt.figure(figsize=(3,3))
        plt.imshow(tf.squeeze(relu), cmap='gray')
        plt.axis('off')
        plt.title('ReLU Activation')
        plt.show()
        
        pooled = tf.nn.pool(relu, window_shape=(2,2), pooling_type='MAX', strides=(2,2), padding='SAME')
        plt.figure(figsize=(3,3))
        plt.imshow(tf.squeeze(pooled), cmap='gray')
        plt.axis('off')
        plt.title('Max Pooling')
        plt.show()
        
    else:
        print(f"File not found: {image_path}")

if processed_images:
    batch_images = tf.stack(processed_images, axis=0)  # shape: [5,28,28,1]
    predictions = model.predict(batch_images)
    
    plt.figure(figsize=(12,4))
    for i, img_path in enumerate(valid_image_files):
        plt.subplot(1, len(valid_image_files), i+1)
        plt.imshow(batch_images[i].numpy().squeeze(), cmap='gray')
        plt.title(f'Pred: {np.argmax(predictions[i])}')
        plt.axis('off')
    plt.show()
