from PIL import Image
import math
import numpy as np

"""
Get and use the functions associated with gaussconvolve2d that you used in the last HW02.
获取并使用上一次作业中使用的与高斯卷积相关的函数。
"""

# Gaussian kernel generation for 1D
# 一维高斯核生成
def gauss1d(sigma):
    size = int(math.ceil(6 * sigma))
    x = np.arange(-size//2, size//2 + 1)
    kernel = np.exp(-x**2 / (2 * sigma**2))
    kernel /= np.sum(kernel)
    return kernel

# Gaussian kernel generation for 2D
# 二维高斯核生成
def gauss2d(sigma):
    size = int(math.ceil(6 * sigma))
    x, y = np.meshgrid(np.arange(-size//2, size//2 + 1), np.arange(-size//2, size//2 + 1))
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel /= np.sum(kernel)
    return kernel

# 2D convolution
# 二维卷积
def convolve2d(array, kernel):
    kernel_size = len(kernel)
    image_height, image_width = array.shape
    padded_array = np.pad(array, ((kernel_size//2, kernel_size//2), (kernel_size//2, kernel_size//2)), mode='edge')
    result = np.zeros_like(array)

    for i in range(image_height):
        for j in range(image_width):
            result[i, j] = np.sum(padded_array[i:i+kernel_size, j:j+kernel_size] * kernel)

    return result

# Gaussian blur applied to an image
# 高斯模糊应用于图像
def gaussconvolve2d(array, sigma):
    kernel = gauss2d(sigma)
    return convolve2d(array, kernel)

# Reduce noise in the image
# 减少图像中的噪声
def reduce_noise(img):
    grayscale_img = img.convert("L")
    grayscale_array = np.array(grayscale_img, dtype=np.float32)
    filtered_array = gaussconvolve2d(grayscale_array, 1.6)
    return filtered_array

# Sobel filters for edge detection
# Sobel滤波器用于边缘检测
def sobel_filters(img):
    """ 
    Returns gradient magnitude and direction of input img.
    返回输入图像的梯度幅值和方向。

    Args:
        img: Grayscale image. Numpy array of shape (H, W).
        img：灰度图像。形状为（H，W）的Numpy数组。

    Returns:
        G: Magnitude of gradient at each pixel in img.
            Numpy array of shape (H, W).
        G：图像中每个像素的梯度幅值。形状为（H，W）的Numpy数组。
        theta: Direction of gradient at each pixel in img.
            Numpy array of shape (H, W).
        theta：图像中每个像素的梯度方向。形状为（H，W）的Numpy数组。

    Hints:
        - Use np.hypot and np.arctan2 to calculate square root and arctan
        - 使用np.hypot和np.arctan2来计算平方根和反正切
    """

    # Sobel operators for detecting edges in x and y directions
    # Sobel算子用于检测x和y方向上的边缘
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])

    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])

    # Apply convolution to get gradients in x and y directions
    # 应用卷积以获取x和y方向上的梯度
    Ix = convolve2d(img, sobel_x)
    Iy = convolve2d(img, sobel_y)

    # Compute gradient magnitude and direction
    # 计算梯度幅值和方向
    G = np.hypot(Ix, Iy)  # Magnitude
    theta = np.arctan2(Iy, Ix)  # Direction

    # Map gradient magnitude to range [0, 255]
    # 将梯度幅值映射到范围[0, 255]
    G = np.interp(G, (G.min(), G.max()), (0, 255))

    return Ix, Iy, G, theta

# Non-maximum suppression to thin edges
# 非极大值抑制以细化边缘
def non_max_suppression(G, theta):
    """ 
    Performs non-maximum suppression.

    Args:
        G: Gradient magnitude image with shape of (H, W).
        G：形状为（H，W）的梯度幅值图像。
        theta: Direction of gradients with shape of (H, W).
        theta：形状为（H，W）的梯度方向。

    Returns:
        res: Non-maxima suppressed image.
        res：非极大值抑制图像。
    """
  
    H, W = G.shape
    res = np.zeros((H, W))

    theta *= 180 / np.pi
    theta[theta < -135] += 180
    theta[theta > 135] -= 180

    for i in range(1, H - 1):
        for j in range(1, W - 1):
            direction = theta[i, j]

            if (-22.5 <= direction < 22.5) or (157.5 <= direction <= 180) or (-157.5 <= direction < -180):
                q = G[i, j+1]
                r = G[i, j-1]
            elif (22.5 <= direction < 67.5) or (-112.5 <= direction < -67.5):
                q = G[i+1, j-1]
                r = G[i-1, j+1]
            elif (67.5 <= direction < 112.5) or (-67.5 <= direction < -22.5):
                q = G[i+1, j]
                r = G[i-1, j]
            elif (-112.5 <= direction < -67.5) or (22.5 <= direction < 67.5):
                q = G[i+1, j+1]
                r = G[i-1, j-1]

            if (G[i, j] >= q) and (G[i, j] >= r):
                res[i, j] = G[i, j]

    return res

# Double thresholding to identify strong and weak edges
# 双阈值处理以识别强边和弱边
def double_thresholding(img):
    """ 
    Args:
        img: Numpy array of shape (H, W) representing NMS edge response.
        img：形状为（H，W）的Numpy数组，表示非极大值抑制的边缘响应。

    Returns:
        res: Double-thresholded image.
        res：双阈值处理后的图像。
    """
    # Compute difference between maximum and minimum intensity
    # 计算最大强度和最小强度之间的差异
    diff = np.max(img) - np.min(img)

    # Calculate high and low threshold values
    # 计算高和低阈值
    Thigh = np.min(img) + diff * 0.15
    Tlow = np.min(img) + diff * 0.03

    # Initialize result array
    # 初始化结果数组
    res = np.zeros_like(img)

    # Classify pixels into strong, weak, and non-relevant based on thresholds
    # 根据阈值将像素分类为强、弱和不相关的像素
    strong_pixels = img >= Thigh
    weak_pixels = (img >= Tlow) & (img < Thigh)

    # Map pixel values accordingly
    # 相应地映射像素值
    res[strong_pixels] = 255
    res[weak_pixels] = 80

    return res

# Depth-first search to link weak edges to strong edges
# 深度优先搜索以将弱边连接到强边
def dfs(img, res, i, j, visited=[]):
    # calling dfs on (i, j) coordinate imply that
    #   1. the (i, j) is strong edge
    #   2. the (i, j) is weak edge connected to a strong edge
    # In case 2, it meets the condition to be a strong edge
    # therefore, change the value of the (i, j) which is weak edge to 255 which is strong edge
    # 在(i, j)坐标上调用dfs意味着
    #   1. (i, j)是强边缘
    #   2. (i, j)是连接到强边缘的弱边缘
    # 对于情况2，它满足成为强边缘的条件
    # 因此，将弱边缘的值更改为255，即强边缘
    res[i, j] = 255

    # mark the visitation
    # 标记访问
    visited.append((i, j))

    # examine (i, j)'s 8 neighbors
    # call dfs recursively if there is a weak edge
    # 检查(i, j)的8个相邻像素
    # 如果存在弱边缘，则递归调用dfs
    for ii in range(i-1, i+2) :
        for jj in range(j-1, j+2) :
            if (img[ii, jj] == 80) and ((ii, jj) not in visited) :
                dfs(img, res, ii, jj, visited)

# Hysteresis to link weak edges connected to strong edges
# 绑定连接到强边缘的弱边缘的环境
def hysteresis(img):
    """ 
    Find weak edges connected to strong edges and link them.
    Iterate over each pixel in strong_edges and perform depth first
    search across the connected pixels in weak_edges to link them.
    Here we consider a pixel (a, b) is connected to a pixel (c, d)
    if (a, b) is one of the eight neighboring pixels of (c, d).
    
    Args:
        img: Numpy array of shape (H, W) representing NMS edge response.
        img：形状为（H，W）的Numpy数组，表示非极大值抑制的边缘响应。
    
    Returns:
        res: Hysteresised image.
        res：环境图像。
    """
    # Initialize result array
    # 初始化结果数组
    res = np.zeros_like(img)

    # Find indices of strong and weak pixels
    # 查找强像素和弱像素的索引
    strong_indices = np.argwhere(img == 255)
    weak_indices = np.argwhere(img == 80)

    # Iterate over strong pixels and perform DFS to connect weak pixels
    # 遍历强像素并执行DFS以连接弱像素
    for i, j in strong_indices:
        # Call DFS for each strong pixel
        # 对每个强像素调用DFS
        dfs(img, res, i, j)

    return res

def main():
    # Load the RGB image
    # 加载RGB图像
    RGB_img = Image.open('./iguana.bmp')

    # Reduce noise in the image
    # 减少图像中的噪声
    noise_reduced_img = reduce_noise(RGB_img)
    Image.fromarray(noise_reduced_img.astype('uint8')).show()

    # Calculate gradients
    # 计算梯度
    Ix, Iy, g, theta = sobel_filters(noise_reduced_img
    Image.fromarray(Ix.astype('uint8')).show()
    Image.fromarray(Iy.astype('uint8')).show()                                 
    Image.fromarray(g.astype('uint8')).show()
    Image.fromarray(theta.astype('uint8')).show()

    # Perform non-maximum suppression
    # 执行非极大值抑制
    non_max_suppression_img = non_max_suppression(g, theta)
    Image.fromarray(non_max_suppression_img.astype('uint8')).show()

    # Perform double thresholding
    # 执行双阈值处理
    double_threshold_img = double_thresholding(non_max_suppression_img)
    Image.fromarray(double_threshold_img.astype('uint8')).show()

    # Perform hysteresis
    # 执行环境
    hysteresis_img = hysteresis(double_threshold_img)
    Image.fromarray(hysteresis_img.astype('uint8')).show()

main()
