import tensorflow as tf
import os
import matplotlib.pyplot as plt

image_files = [
    'images.jpg',
    'aic-home (1).jpg',
    'jpegls-home (1).jpg',
    'Un_super_paysage.jpg',
    'what-is-jpeg-format (1).jpg'
]

# Define the edge detection kernel (Laplacian)
kernel = tf.constant([
    [-1., -1., -1.],
    [-1.,  8., -1.],
    [-1., -1., -1.]
], dtype=tf.float32)
kernel = tf.reshape(kernel, [3, 3, 1, 1])  # [height, width, in_channels, out_channels]
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
        
        # Apply convolution (edge detection)
        conv = tf.nn.conv2d(image, filters=kernel, strides=1, padding='SAME')
        
        # Apply ReLU activation
        relu = tf.nn.relu(conv)
        print("Original ReLU shape:", relu.shape)
        
        # Max pooling
        pooled = tf.nn.pool(
            input=relu,
            window_shape=(2, 2),
            pooling_type='MAX',
            strides=(2, 2),
            padding='SAME'
        )
        print("Pooled shape:", pooled.shape)
        
        # For display, remove batch and channel dimensions
        pooled_display = tf.squeeze(pooled)
        print("Shape for display:", pooled_display.shape)
        
        # Display
        plt.figure(figsize=(5,5))
        plt.imshow(pooled_display.numpy(), cmap='gray')
        plt.axis('off')
        plt.title(f'After Max Pooling: {image_path}')
        plt.show()
        
    else:
        print(f"File not found: {image_path}")
