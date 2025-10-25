import tensorflow as tf
import os
import matplotlib.pyplot as plt

# List of image files
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
kernel = tf.reshape(kernel, [3, 3, 1, 1])  # shape: [height, width, in_channels, out_channels]

for image_path in image_files:
    if os.path.exists(image_path):
        # Read, decode, grayscale, resize
        image = tf.io.read_file(image_path)
        image = tf.io.decode_jpeg(image, channels=1)
        image = tf.image.resize(image, [300, 300])
        # Convert to float32
        image = tf.image.convert_image_dtype(image, tf.float32)
        # Add batch dimension
        image = tf.expand_dims(image, axis=0)
        print(f"Image shape ready for CNN ({image_path}):", image.shape)
        
        conv = tf.nn.conv2d(image, filters=kernel, strides=1, padding='SAME')
        
        # Display convolution result
        plt.figure(figsize=(5,5))
        plt.imshow(tf.squeeze(conv), cmap='gray')
        plt.axis('off')
        plt.title(f'Edges Detected: {image_path}')
        plt.show()
        
    else:
        print(f"File not found: {image_path}")
