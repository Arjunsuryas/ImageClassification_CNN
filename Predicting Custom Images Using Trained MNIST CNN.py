import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

# List of your 5 images
image_files = [
    'images.jpg',
    'aic-home (1).jpg',
    'jpegls-home (1).jpg',
    'Un_super_paysage.jpg',
    'what-is-jpeg-format (1).jpg'
]

# Preprocess images for CNN input (grayscale, resize to 28x28, normalize)
processed_images = []
valid_image_files = []

for image_path in image_files:
    if os.path.exists(image_path):
        image = tf.io.read_file(image_path)
        image = tf.io.decode_jpeg(image, channels=1)      # grayscale
        image = tf.image.resize(image, [28, 28])         # resize to MNIST size
        image = tf.image.convert_image_dtype(image, tf.float32)  # normalize
        processed_images.append(image)
        valid_image_files.append(image_path)
    else:
        print(f"File not found: {image_path}")

# Convert list to batch tensor
if processed_images:
    batch_images = tf.stack(processed_images, axis=0)  # shape: [5,28,28,1]
    print("Batch shape ready for CNN:", batch_images.shape)

    # Predict using trained CNN model
    predictions = model.predict(batch_images)

    # Display images with predicted labels
    plt.figure(figsize=(10,6))
    for i, img_path in enumerate(valid_image_files):
        plt.subplot(1, len(valid_image_files), i+1)
        plt.imshow(batch_images[i].numpy().squeeze(), cmap='gray')
        plt.title(f"Pred: {np.argmax(predictions[i])}")
        plt.axis('off')
    plt.show()
