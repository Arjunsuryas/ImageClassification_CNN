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
processed_images = []

for image_path in image_files:
    if os.path.exists(image_path):
        # Read and decode image as grayscale
        image = tf.io.read_file(image_path)
        image = tf.io.decode_jpeg(image, channels=1)  # grayscale
        # Resize to 300x300
        image = tf.image.resize(image, [300, 300])
        # Convert to float32 [0,1]
        image = tf.image.convert_image_dtype(image, tf.float32)
        # Add batch dimension
        image = tf.expand_dims(image, axis=0)
        print(f"Image shape ready for CNN ({image_path}):", image.shape)
        # Store in list
        processed_images.append(image)
        
        # Optional: display image
        plt.figure(figsize=(5,5))
        plt.imshow(tf.squeeze(image), cmap='gray')
        plt.axis('off')
        plt.title(f'Prepared Image: {image_path}')
        plt.show()
    else:
        print(f"File not found: {image_path}")

if processed_images:
    batch_images = tf.concat(processed_images, axis=0)
    print("Batch shape ready for CNN:", batch_images.shape)
