import multiprocessing
import scipy.ndimage as sc  # Fixed import
import numpy as np
import matplotlib.pyplot as plt

# Using tensorflow just to grab the dataset easily.
# If you don't have tf, you can use sklearn.datasets.fetch_openml('mnist_784')
try:
    from tensorflow.keras.datasets import mnist
except ImportError:
    print("This script requires tensorflow to load the MNIST dataset easily.")
    print("pip install tensorflow matplotlib scipy numpy")
    exit()


from preprocessor import Worker


# --- Test Script ---

def show_augmentation_test():
    # 1. Load Data
    print("Loading MNIST data...")
    (x_train, y_train), _ = mnist.load_data()

    # 2. Select a random image and Normalize (0-255 -> 0.0-1.0)
    # We select a '5' or similar distinctive digit
    idx = np.where(y_train == 0)[0][0]
    original_image = x_train[idx].astype('float32') / 255.0

    # Ensure it is 28x28 for geometric ops
    if original_image.shape != (28, 28):
        original_image = original_image.reshape(28, 28)

    # 3. Instantiate Worker (dummy args for testing static methods)
    worker = Worker(None, None, None, None)

    # 4. Perform Augmentations
    # A. Rotation (e.g., 30 degrees)
    img_rot = worker.rotate(original_image, 30)

    # B. Shift (e.g., 5 pixels right, 5 down)
    img_shift = worker.shift(original_image, 5, 5)

    # C. Noise (Add random noise)
    img_noise = worker.add_noise(original_image, 0.3)

    # D. Skew (Tilt the image)
    img_skew = worker.skew(original_image, 0.3)

    # E. Combined (The process_image pipeline)
    img_combined = worker.process_image(original_image)

    # 5. Plot Results
    fig, axes = plt.subplots(1, 6, figsize=(18, 4))
    titles = ["Original", "Rotated (30°)", "Shifted (5,5)", "Noise (0.3)", "Skewed (0.4)", "All Combined"]
    images = [original_image, img_rot, img_shift, img_noise, img_skew, img_combined]

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap='gray')
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()
    print("Displaying results...")
    plt.show()


if __name__ == "__main__":
    show_augmentation_test()