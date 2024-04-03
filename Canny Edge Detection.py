from PIL import Image
import math
import numpy as np

def gauss1d(sigma):
    size = int(math.ceil(6 * sigma))
    x = np.arange(-size//2, size//2 + 1)
    kernel = np.exp(-x**2 / (2 * sigma**2))
    kernel /= np.sum(kernel)
    return kernel

def gauss2d(sigma):
    size = int(math.ceil(6 * sigma))
    x, y = np.meshgrid(np.arange(-size//2, size//2 + 1), np.arange(-size//2, size//2 + 1))
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel /= np.sum(kernel)
    return kernel

def convolve2d(array, filter):
    kernel_size = len(filter)
    image_height, image_width = array.shape
    padded_array = np.pad(array, ((kernel_size//2, kernel_size//2), (kernel_size//2, kernel_size//2)), mode='edge')
    result = np.zeros_like(array)

    for i in range(image_height):
        for j in range(image_width):
            result[i, j] = np.sum(padded_array[i:i+kernel_size, j:j+kernel_size] * filter)

    return result

def gaussconvolve2d(array, sigma):
    kernel = gauss2d(sigma)
    return convolve2d(array, kernel)

def reduce_noise(img):
    grayscale_img = img.convert("L")
    grayscale_array = np.array(grayscale_img, dtype=np.float32)
    filtered_array = gaussconvolve2d(grayscale_array, 1.6)
    return filtered_array

def sobel_filters(img):
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    Ix = convolve2d(img, sobel_x)
    Iy = convolve2d(img, sobel_y)

    G = np.hypot(Ix, Iy)
    theta = np.arctan2(Iy, Ix)

    G = (G / np.max(G)) * 255

    return G, theta

def non_max_suppression(G, theta):
    H, W = G.shape
    res = np.zeros((H, W))

    theta *= 180 / np.pi
    theta[theta < 0] += 180

    for i in range(1, H - 1):
        for j in range(1, W - 1):
            direction = theta[i, j]

            if (0 <= direction < 22.5) or (157.5 <= direction <= 180) or (67.5 <= direction < 112.5):
                q = G[i, j+1]
                r = G[i, j-1]
            elif (22.5 <= direction < 67.5) or (112.5 <= direction < 157.5):
                q = G[i+1, j-1]
                r = G[i-1, j+1]
            elif (67.5 <= direction < 112.5) or (157.5 <= direction < 202.5):
                q = G[i+1, j]
                r = G[i-1, j]
            elif (112.5 <= direction < 157.5) or (22.5 <= direction < 67.5):
                q = G[i+1, j+1]
                r = G[i-1, j-1]

            if (G[i, j] >= q) and (G[i, j] >= r):
                res[i, j] = G[i, j]

    return res

def dfs(weak_edges, strong_edges, i, j, visited):
    stack = [(i, j)]
    visited.add((i, j))

    while stack:
        curr_i, curr_j = stack.pop()
        strong_edges[curr_i, curr_j] = 255

        for ii in range(curr_i-1, curr_i+2):
            for jj in range(curr_j-1, curr_j+2):
                if weak_edges[ii, jj] == 80 and (ii, jj) not in visited:
                    stack.append((ii, jj))
                    visited.add((ii, jj))

def double_thresholding(img):
    diff = np.max(img) - np.min(img)
    high_threshold = np.min(img) + diff * 0.15
    low_threshold = np.min(img) + diff * 0.03

    strong_edges = np.where(img >= high_threshold, 255, 0)
    weak_edges = np.where((img >= low_threshold) & (img < high_threshold), 80, 0)

    H, W = weak_edges.shape
    visited = set()

    for i in range(1, H - 1):
        for j in range(1, W - 1):
            if weak_edges[i, j] == 80 and (i, j) not in visited:
                dfs(weak_edges, strong_edges, i, j, visited)

    return strong_edges

def hysteresis(array):
    strong_edges = np.where(array == 255)
    weak_edges = np.where(array == 80)

    for i, j in zip(strong_edges[0], strong_edges[1]):
        dfs(array, array, i, j)

    return array

def main():
    RGB_img = Image.open('./iguana.bmp')

    noise_reduced_img = reduce_noise(RGB_img)
    Image.fromarray(noise_reduced_img.astype('uint8')).show()
    
    g, theta = sobel_filters(noise_reduced_img)
    Image.fromarray(g.astype('uint8')).show()
    Image.fromarray(theta.astype('uint8')).show()

    non_max_suppression_img = non_max_suppression(g, theta)
    Image.fromarray(non_max_suppression_img.astype('uint8')).show()

    double_threshold_img = double_thresholding(non_max_suppression_img)
    Image.fromarray(double_threshold_img.astype('uint8')).show()

main()
