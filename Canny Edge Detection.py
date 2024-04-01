from PIL import Image
import math
import numpy as np

"""
Get and use the functions associated with gaussconvolve2d that you used in the last HW02.
"""
def gauss1d(sigma):
    size = int(2 * sigma + 1)
    kernel = np.zeros(size)
    center = size // 2
    const = 1 / (sigma * math.sqrt(2 * math.pi))

    for i in range(size):
        kernel[i] = const * math.exp(-((i - center) ** 2) / (2 * sigma ** 2))

    return kernel / np.sum(kernel)

def gauss2d(sigma):
    size = int(2 * sigma + 1)
    kernel = np.zeros((size, size))
    center = size // 2
    const = 1 / (2 * math.pi * sigma ** 2)

    for i in range(size):
        for j in range(size):
            kernel[i, j] = const * math.exp(-((i - center) ** 2 + (j - center) ** 2) / (2 * sigma ** 2))

    return kernel / np.sum(kernel)

def convolve2d(array, kernel):
    m, n = array.shape
    km, kn = kernel.shape
    kh, kw = km // 2, kn // 2
    result = np.zeros_like(array)

    for i in range(kh, m - kh):
        for j in range(kw, n - kw):
            result[i, j] = np.sum(array[i-kh:i+kh+1, j-kw:j+kw+1] * kernel)

    return result

def gaussconvolve2d(array, sigma):
    kernel = gauss2d(sigma)
    return convolve2d(array, kernel)


def reduce_noise(img):
    """ 返回灰度高斯滤波后的图像，sigma=1.6
    Args:
        img: RGB图像。形状为(H, W, 3)的Numpy数组。
    Returns:
        res: 灰度高斯滤波后的图像 (H, W)。
    """
    grayscale_img = img.convert('L')  # 转换为灰度图像
    grayscale_array = np.array(grayscale_img, dtype=np.float32)  # 转换为numpy数组
    filtered_array = gaussconvolve2d(grayscale_array, 1.6)  # 应用高斯模糊
    return filtered_array

def sobel_filters(img):
    """ 返回图像的梯度幅度和方向。
    Args:
        img: 灰度图像。形状为(H, W)的Numpy数组。
    Returns:
        G: 图像中每个像素的梯度幅度。
            形状为(H, W)的Numpy数组。
        theta: 图像中每个像素的梯度方向。
            形状为(H, W)的Numpy数组。
    """
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    Ix = convolve2d(img, sobel_x)
    Iy = convolve2d(img, sobel_y)

    G = np.sqrt(Ix ** 2 + Iy ** 2)
    theta = np.arctan2(Iy, Ix)

    return (G, theta)

def non_max_suppression(G, theta):
    """ 执行非极大值抑制。
    This function performs non-maximum suppression along the direction
    of gradient (theta) on the gradient magnitude image (G).
    Args:
        G: gradient magnitude image with shape of (H, W).
        theta: direction of gradients with shape of (H, W).
    Returns:
        res: non-maxima suppressed image.
    """
    H, W = G.shape
    res = np.zeros((H, W))

    for i in range(1, H - 1):
        for j in range(1, W - 1):
            direction = theta[i, j] * (180 / np.pi)

            if (0 <= direction < 22.5) or (157.5 <= direction <= 180) or (-22.5 <= direction < 0) or (-180 <= direction < -157.5):
                q = G[i, j+1]
                r = G[i, j-1]
            elif (22.5 <= direction < 67.5) or (-157.5 <= direction < -112.5):
                q = G[i+1, j-1]
                r = G[i-1, j+1]
            elif (67.5 <= direction < 112.5) or (-112.5 <= direction < -67.5):
                q = G[i+1, j]
                r = G[i-1, j]
            elif (112.5 <= direction < 157.5) or (-67.5 <= direction < -22.5):
                q = G[i-1, j-1]
                r = G[i+1, j+1]

            if (G[i, j] >= q) and (G[i, j] >= r):
                res[i, j] = G[i, j]

    return res

def double_thresholding(img):
    """ 
    Args:
        img: numpy array of shape (H, W) representing NMS edge response.
    Returns:
        res: double_thresholded image.
    """
    diff = np.max(img) - np.min(img)
    high_threshold = np.min(img) + diff * 0.15
    low_threshold = np.min(img) + diff * 0.03

    strong_edges = np.where(img >= high_threshold, 255, 0)
    weak_edges = np.where((img >= low_threshold) & (img < high_threshold), 80, 0)

    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            if strong_edges[i, j] == 255:
                dfs(img, strong_edges, i, j)

    return strong_edges

def dfs(img, res, i, j, visited=[]):
    res[i, j] = 255
    visited.append((i, j))

    for ii in range(i-1, i+2):
        for jj in range(j-1, j+2):
            if (img[ii, jj] == 80) and ((ii, jj) not in visited):
                dfs(img, res, ii, jj, visited)

def hysteresis(img):
    """ Find weak edges connected to strong edges and link them.
    Iterate over each pixel in strong_edges and perform depth first
    search across the connected pixels in weak_edges to link them.
    Here we consider a pixel (a, b) is connected to a pixel (c, d)
    if (a, b) is one of the eight neighboring pixels of (c, d).
    Args:
        img: numpy array of shape (H, W) representing NMS edge response.
    Returns:
        res: hysteresised image.
    """
    res = np.zeros_like(img)
    strong_indices = np.where(img == 255)

    for i, j in zip(strong_indices[0], strong_indices[1]):
        dfs(img, res, i, j)

    return res

def main():
    # 这部分已经给出，不需要修改
    RGB_img = Image.open('./iguana.bmp')

    noise_reduced_img = reduce_noise(RGB_img)
    Image.fromarray(noise_reduced_img.astype('uint8')).save('./iguana_blurred.bmp', 'BMP')
    
    g, theta = sobel_filters(noise_reduced_img)
    Image.fromarray(g.astype('uint8')).save('./iguana_sobel_gradient.bmp', 'BMP')
    Image.fromarray(theta.astype('uint8')).save('./iguana_sobel_theta.bmp', 'BMP')

    non_max_suppression_img = non_max_suppression(g, theta)
    Image.fromarray(non_max_suppression_img.astype('uint8')).save('./iguana_non_max_suppression.bmp', 'BMP')

    double_threshold_img = double_thresholding(non_max_suppression_img)
    Image.fromarray(double_threshold_img.astype('uint8')).save('./iguana_double_thresholding.bmp', 'BMP')

    hysteresis_img = hysteresis(double_threshold_img)
    Image.fromarray(hysteresis_img.astype('uint8')).save('./iguana_hysteresis.bmp', 'BMP')


