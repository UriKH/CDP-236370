import numpy as np
import matplotlib.pyplot as plt
from filters import correlation_numba, load_image


def run_sobel_variations():
    image = load_image()

    # 1. Standard Sobel
    k_sobel = np.array([[1, 0, -1],
                        [2, 0, -2],
                        [1, 0, -1]])

    # 2. Kernel 1 (Stronger center weight - likely Scharr-like)
    k1 = np.array([[3, 0, -3],
                   [10, 0, -10],
                   [3, 0, -3]])

    # 3. Kernel 2 (Tall 5x3 kernel - smoothing vertical noise)
    # Note: The PDF OCR shows row 4 as [1, 0, -2], but standard kernels are usually symmetric.
    # We will use the values exactly as they appear in the PDF text just in case.
    k2 = np.array([[1, 0, -1],
                   [2, 0, -2],
                   [2, 0, -2],
                   [1, 0, -2],  # Note the -2 here based on PDF source
                   [1, 0, -1]])

    # 4. Kernel 3 (All 1s - Box Blur/Sum)
    k3 = np.array([[1, 1, 1],
                   [1, 1, 1],
                   [1, 1, 1]])

    kernels = [("Standard Sobel", k_sobel), ("Kernel 1", k1), ("Kernel 2", k2), ("Kernel 3", k3)]

    plt.figure(figsize=(12, 8))

    for idx, (name, kern) in enumerate(kernels):
        # Calculate Magnitude
        gx = correlation_numba(kern, image)
        gy = correlation_numba(kern.T, image)
        magnitude = np.sqrt(gx ** 2 + gy ** 2)

        # Normalize for display (0-255)
        magnitude = (magnitude / magnitude.max()) * 255

        plt.subplot(2, 2, idx + 1)
        plt.imshow(magnitude, cmap='gray')
        plt.title(name)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig("sobel_variations.png")
    plt.show()


if __name__ == "__main__":
    run_sobel_variations()