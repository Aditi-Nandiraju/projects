import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to rotate the image
def rotate(image):
    (height, width) = image.shape[:2]
    angle = float(input("Enter the angle to rotate the image: "))
    scale = 1
    
    angle_rad = np.deg2rad(angle)
    cos_angle = np.cos(angle_rad) * scale
    sin_angle = np.sin(angle_rad) * scale

    rotation_matrix = np.array([
        [cos_angle, -sin_angle, (1 - cos_angle) * (width / 2) + sin_angle * (height / 2)],
        [sin_angle, cos_angle, (1 - cos_angle) * (height / 2) - sin_angle * (width / 2)],
        [0, 0, 1]
    ])

    output = np.zeros_like(image)
    inverse_matrix = np.linalg.inv(rotation_matrix)

    for y in range(output.shape[0]):
        for x in range(output.shape[1]):
            coord = np.dot(inverse_matrix, [x, y, 1])
            orig_x, orig_y = int(coord[0]), int(coord[1])
            if 0 <= orig_x < width and 0 <= orig_y < height:
                output[y, x] = image[orig_y, orig_x]

    plt.figure(figsize=(10, 6))

    # Original image
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Rotated image
    plt.subplot(1, 2, 2)
    plt.title(f"Rotated Image (Angle: {angle}°)")
    plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))

    plt.tight_layout()
    plt.show()

# Function to scale the image
def scalling(image):
    scale_x = float(input("Enter the scaling factor for width: "))
    scale_y = float(input("Enter the scaling factor for height: "))
    
    (height, width) = image.shape[:2]

    scaling_matrix = np.array([
        [scale_x, 0, 0],
        [0, scale_y, 0],
        [0, 0, 1]
    ])

    new_width = int(width * scale_x)
    new_height = int(height * scale_y)

    scaled_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)

    inverse_scaling_matrix = np.linalg.inv(scaling_matrix)
    for y in range(new_height):
        for x in range(new_width):
            original_coords = np.dot(inverse_scaling_matrix, [x, y, 1])
            orig_x, orig_y = int(original_coords[0]), int(original_coords[1])
            if 0 <= orig_x < width and 0 <= orig_y < height:
                scaled_image[y, x] = image[orig_y, orig_x]

    plt.figure(figsize=(10, 6))

    # Original image
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Scaled image
    plt.subplot(1, 2, 2)
    plt.title(f"Scaled Image (ScaleX: {scale_x}, ScaleY: {scale_y})")
    plt.imshow(cv2.cvtColor(scaled_image, cv2.COLOR_BGR2RGB))

    plt.tight_layout()
    plt.show()

# Function to blur the image
def blur(image):
    kernel_size = int(input("Enter the size of the kernel to blur: "))
    if kernel_size % 2 == 0 or kernel_size <= 0:
        print("Error: Kernel size must be a positive odd number.")
        return

    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size ** 2)

    height, width, channels = image.shape
    pad = kernel_size // 2
    padded_image = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='constant', constant_values=0)

    blurred_image = np.zeros_like(image)

    for c in range(channels): 
        for y in range(height):
            for x in range(width):
                region = padded_image[y:y + kernel_size, x:x + kernel_size, c]
                blurred_image[y, x, c] = np.sum(region * kernel)

    plt.figure(figsize=(10, 6))

    # Original image
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  
    # Blurred image
    plt.subplot(1, 2, 2)
    plt.title(f"Blurred Image (Kernel Size: {kernel_size}x{kernel_size})")
    plt.imshow(cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB))  

    plt.tight_layout()
    plt.show()


def main():
    image_path = "D:\\python codes\\sample.jpg"
    image = cv2.imread(image_path)

    print("Select from the following actions:")
    print("1. Rotation")
    print("2. Scaling")
    print("3. Blurring")
    x = int(input("Enter your choice: "))
    if x == 1:
        rotate(image)
    if x == 2:
        scalling(image)
    if x == 3:
        blur(image)
    else:
        print("Invalid ")

if __name__ == "__main__":
    main()
