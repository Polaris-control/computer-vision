import math
import numpy as np
from PIL import Image

def boxfilter(n):
    assert n % 2 != 0, "Dimension must be odd"
    box_filter = np.full((n,n),1/(n*n))
    return box_filter

# Test cases
print(boxfilter(3))
print(boxfilter(5))
print(boxfilter(7))

def gauss1d(sigma):
    # Calculate the length of the filter
    length = int(np.ceil(6 * sigma))
    if length % 2 == 0:
        length += 1
    
    # Generate 1D array of x values
    x = np.arange(-length // 2, length // 2 + 1)
    
    # Compute the Gaussian function
    gaussian_filter = np.exp(-x ** 2 / (2 * sigma ** 2))
    
    # Normalize the filter values
    normalized_filter = gaussian_filter / np.sum(gaussian_filter)
    
    return normalized_filter

# Test cases
sigmas = [0.3, 0.5, 1, 2]
for sigma in sigmas:
    print(f"Sigma = {sigma}:")
    print(gauss1d(sigma))
    print()


def gauss2d(sigma):
    # Generate 1D Gaussian filter
    gaussian_1d = gauss1d(sigma)
    
    # Generate 2D Gaussian filter using outer product
    gaussian_2d = np.outer(gaussian_1d, gaussian_1d)
    
    return gaussian_2d

# Test cases
sigmas = [0.5, 1]
for sigma in sigmas:
    print(f"Sigma = {sigma}:")
    print(gauss2d(sigma))
    print()

def convolve2d(array, filter):
    # Get dimensions of input array and filter
    array_height, array_width = array.shape
    filter_height, filter_width = filter.shape
    
    # Calculate padding needed for convolution
    pad_height = filter_height // 2
    pad_width = filter_width // 2
    
    # Create zero-padded array
    padded_array = np.pad(array, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')
    
    # Initialize result array
    result = np.zeros_like(array, dtype=np.float32)
    
    # Perform convolution
    for i in range(array_height):
        for j in range(array_width):
            # Extract neighborhood
            neighborhood = padded_array[i:i+filter_height, j:j+filter_width]
            # Convolution
            result[i, j] = np.sum(neighborhood * filter)
    
    return result

def gaussconvolve2d(array, sigma):
    # Generate 2D Gaussian filter
    filter = gauss2d(sigma)
    
    # Perform convolution
    result = convolve2d(array, filter)
    
    return result

# Load image and convert to grayscale numpy array
def load_image(image_path):
    image = Image.open(image_path)
    image_gray = image.convert("L")
    image_array = np.array(image_gray, dtype=np.float32)
    return image_array

def save_image(image_array, file_name):
    # Convert array back to unsigned integer format
    image_array = image_array.astype(np.uint8)
    # Convert array back to image
    image = Image.fromarray(image_array)
    # Save the image
    image.save(file_name)
    # Display the image
    image.show()

# Test gaussconvolve2d with a sigma of 3 on the image
image_array = load_image("3a_lion.bmp")
sigma = 3
filtered_image_array = gaussconvolve2d(image_array, sigma)

# Save and display the original and filtered images

save_image(image_array, f"original_{sigma}_lion.jpg")
save_image(filtered_image_array, f"filtered_{sigma}_lion.jpg")




def part2_1(image_path, sigma):
    # Load the image and convert to numpy array
    image_array = load_image(image_path)
    
    # Apply Gaussian convolution
    blurred_image_array = gaussconvolve2d(image_array, sigma)
    
    # Save the blurred image with the provided file name
    save_image(blurred_image_array, f"blurred_{sigma}_lion.jpg")

# Example usage
image_path = "original_3_lion.jpg"  # Replace with your image file path
sigma = 5  # Choose an appropriate sigma value
part2_1(image_path, sigma)


def part2_2(image_path, sigma):
    # Load the image and convert to numpy array
    image_array = load_image(image_path)
    
    # Apply Gaussian convolution
    blurred_image_array = gaussconvolve2d(image_array, sigma)
    
    # Compute high frequency image
    high_frequency_array = image_array - blurred_image_array
    
    # Zero-center and scale for visualization
    high_frequency_array = high_frequency_array + 128
    
    # Save the high frequency image with the provided file name
    save_image(high_frequency_array, f"high_frequency_{sigma}_lion.jpg")

# Example usage
image_path = "original_3_lion.jpg"  # Replace with your image file path
sigma = 5  # Choose an appropriate sigma value
part2_2(image_path, sigma)


def part2_3(image_path, sigma):
    # Load the original image
    original_image_array = load_image(image_path)
    
    # Apply Gaussian convolution to obtain low frequency image
    low_frequency_array = gaussconvolve2d(original_image_array, sigma)
    
    # Compute high frequency image without scaling
    high_frequency_array = original_image_array - low_frequency_array
    
    # Add low and high frequency images (per channel)
    composite_image_array = original_image_array + high_frequency_array
    
    # Save the composite image with the provided file name
    save_image(composite_image_array, f"composite_{sigma}_lion.jpg")

# Example usage
image_path = "original_3_lion.jpg"  # Replace with your image file path
sigma = 5  # Choose an appropriate sigma value
part2_3(image_path, sigma)
