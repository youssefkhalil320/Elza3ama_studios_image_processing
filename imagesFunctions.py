import cv2
import numpy as np


# Define preprocessing functions
def grayscale(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image

def blur(image):
    blurred_image = cv2.GaussianBlur(image, (15, 15), 0)
    return blurred_image

def edge_detection(image):
    edges = cv2.Canny(image, 100, 200)
    return edges

def invert_colors(image):
    inverted_image = cv2.bitwise_not(image)
    return inverted_image

def threshold(image):
    _, thresh_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
    return thresh_image

def gray_level_transform(image, alpha=1.0, beta=0.0):
    """
    Apply a simple gray level transformation to the image.
    Formula: new_intensity = alpha * old_intensity + beta
    """
    transformed_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return transformed_image

def negative_transform(image):
    """
    Apply a negative transformation to the image.
    """
    negative_image = 255 - image  # Invert pixel values
    return negative_image

def log_transform(image, c=1):
    """
    Apply a logarithmic transformation to the image.
    """
    log_image = np.log1p(c * image)  # Apply log transformation
    # Scale the values to the range [0, 255]
    log_image = (log_image / np.max(log_image)) * 255
    log_image = np.uint8(log_image)
    return log_image

def power_law_transform(image, gamma=1.0):
    """
    Apply a power law transformation (gamma correction) to the image.
    """
    # Apply gamma correction
    power_law_image = np.power(image / 255.0, gamma)
    # Scale the values back to the range [0, 255]
    power_law_image = np.uint8(power_law_image * 255)
    return power_law_image

def contrast_stretching(image, low=0, high=255):
    """
    Stretch the contrast of an image by mapping pixel values to a new range.

    Args:
        image: A numpy array representing the image.
        low: The minimum value in the output image (default: 0).
        high: The maximum value in the output image (default: 255).

    Returns:
        A numpy array representing the contrast-stretched image.
    """
    # Find the minimum and maximum values in the image
    min_val = np.amin(image)
    max_val = np.amax(image)

    # Check if min and max are the same (no stretch needed)
    if min_val == max_val:
        return image

    # Normalize the pixel values to the range [0, 1]
    normalized = (image - min_val) / (max_val - min_val)

    # Stretch the normalized values to the new range [low, high]
    stretched = normalized * (high - low) + low

    # Convert the stretched values back to the original data type (uint8 for images)
    return np.uint8(stretched * 255)


def intensity_slicing(image, threshold):
    """
    Perform intensity slicing on an image to create a binary image.

    Args:
        image: A numpy array representing the image.
        threshold: The intensity threshold for binarization (default: 128).

    Returns:
        A numpy array representing the binary image after intensity slicing.
    """
    # Create a copy of the image to avoid modifying the original
    sliced_image = image.copy()

    # Apply thresholding
    sliced_image[sliced_image > threshold] = 255  # Set values above threshold to white (255)
    sliced_image[sliced_image <= threshold] = 0  # Set values below or equal to threshold to black (0)

    return sliced_image

def histogram_equalization(image):
    """
    Perform histogram equalization on an image to enhance its contrast.

    Args:
        image: A numpy array representing the image.

    Returns:
        A numpy array representing the image after histogram equalization.
    """
    # Compute histogram of the input image
    hist, _ = np.histogram(image.flatten(), bins=256, range=(0,256))

    # Compute cumulative distribution function (CDF)
    cdf = hist.cumsum()

    # Normalize CDF
    cdf_normalized = cdf * hist.max() / cdf.max()

    # Perform histogram equalization
    equalized_image = np.interp(image.flatten(), range(256), cdf_normalized).reshape(image.shape)

    return equalized_image.astype(np.uint8)


def mean_filter(image, kernel_size=3):
    """
    Apply a mean filter (averaging filter) to the image.

    Args:
        image: A numpy array representing the input image.
        kernel_size: The size of the square kernel (default: 3).

    Returns:
        A numpy array representing the image after applying the mean filter.
    """
    # Define the kernel
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)

    # Apply convolution with the kernel using OpenCV's filter2D function
    filtered_image = cv2.filter2D(image, -1, kernel)

    return filtered_image

def gaussian_filter(image, kernel_size=3, sigma=1):
    """
    Apply a Gaussian filter to the image.

    Args:
        image: A numpy array representing the input image.
        kernel_size: The size of the square kernel (default: 3).
        sigma: The standard deviation of the Gaussian distribution (default: 1).

    Returns:
        A numpy array representing the image after applying the Gaussian filter.
    """
    # Generate Gaussian kernel
    kernel = cv2.getGaussianKernel(kernel_size, sigma)
    kernel = np.outer(kernel, kernel.transpose())

    # Apply convolution with the kernel using OpenCV's filter2D function
    filtered_image = cv2.filter2D(image, -1, kernel)

    return filtered_image

def sobel_filter(image):
    """
    Apply the Sobel filter to the image for edge detection.

    Args:
        image: A numpy array representing the input image.

    Returns:
        A numpy array representing the image after applying the Sobel filter.
    """
    # Apply Sobel filter for horizontal gradient
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    
    # Apply Sobel filter for vertical gradient
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # Combine horizontal and vertical gradients to get the gradient magnitude
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

    # Normalize the gradient magnitude to the range [0, 255]
    gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    return gradient_magnitude

def convert_to_grayscale(image):
    """
    Converts an image to grayscale.

    Args:
        image: A NumPy array representing the image (BGR or RGB format).

    Returns:
        A NumPy array representing the grayscale image.
    """
    # Check if image is already grayscale
    if len(image.shape) == 2:
        return image  # Already grayscale

    # Convert the image to grayscale using OpenCV's BGR2GRAY conversion
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return gray_image


def laplacian_filter(image):
    """
    Apply the Laplacian filter to the grayscale image for edge detection.

    Args:
        image: A numpy array representing the input image.

    Returns:
        A numpy array representing the image after applying the Laplacian filter.
    """
    # Convert the input image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Laplacian filter using OpenCV's Laplacian function
    laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)

    # Convert the output to uint8 and scale to [0, 255]
    laplacian = cv2.normalize(laplacian, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    return laplacian

def min_max_filter(image, kernel_size=3, mode='min'):
    """
    Apply the min-max filter to the image.

    Args:
        image: A numpy array representing the input image.
        kernel_size: The size of the square kernel (default: 3).
        mode: The mode of the filter ('min' or 'max') (default: 'min').

    Returns:
        A numpy array representing the image after applying the min-max filter.
    """
    # Define the kernel
    kernel = np.ones((kernel_size, kernel_size))

    # Apply minimum or maximum filter
    if mode == 'min':
        filtered_image = cv2.erode(image, kernel)
    elif mode == 'max':
        filtered_image = cv2.dilate(image, kernel)
    else:
        raise ValueError("Invalid mode. Mode must be 'min' or 'max'.")

    return filtered_image

def median_filter(image, kernel_size=3):
    """
    Apply the median filter to the image.

    Args:
        image: A numpy array representing the input image.
        kernel_size: The size of the square kernel (default: 3).

    Returns:
        A numpy array representing the image after applying the median filter.
    """
    # Ensure that kernel_size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1

    # Apply median filter using OpenCV's medianBlur function
    filtered_image = cv2.medianBlur(image, kernel_size)

    return filtered_image
