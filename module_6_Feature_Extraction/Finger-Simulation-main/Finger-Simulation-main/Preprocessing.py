import numpy as np
import cv2
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
class Enhancement:
    def __init__(self, num_workers=1):
        self.num_workers = num_workers

    def adjust_gamma_single(self, image, gamma):
        return np.clip(np.power(image / 255.0, gamma) * 255.0, 0, 255).astype(np.uint8)

    def adjust_gamma(self, images, gamma=1.0):
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(executor.map(self.adjust_gamma_single, images, repeat(gamma)))
        return np.array(results)

    def adjust_contrast_single(self, image, alpha, beta):
        return np.clip(alpha * image + beta, 0, 255).astype(np.uint8)

    def adjust_contrast(self, images, alpha=1.0, beta=0.0):
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(executor.map(self.adjust_contrast_single, images, repeat(alpha), repeat(beta)))
        return np.array(results)

    def adjust_brightness_single(self, image, beta):
        return np.clip(image + beta, 0, 255).astype(np.uint8)

    def adjust_brightness(self, images, beta=0.0):
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(executor.map(self.adjust_brightness_single, images, repeat(beta)))
        return np.array(results)
    
    def equalize_histogram_single(self, image):
        flat_image = image.flatten()
        histogram, _ = np.histogram(flat_image, bins=256, range=(0, 256))
        cumulative_histogram = np.cumsum(histogram)
        normalization_factor = 255 / cumulative_histogram[-1]
        mapping = np.round(normalization_factor * cumulative_histogram).astype(np.uint8)
        return mapping[image]
    
    def equalize_histogram(self, images):
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(executor.map(self.equalize_histogram_single, images))
        return results
    
    def blur_single(self, image, kernel_size, blur_type):
        pad_size = kernel_size // 2
        image_padded = np.pad(image, (pad_size, pad_size), mode="constant")
        result = np.zeros_like(image)
        for x in range(pad_size, image.shape[0] + pad_size):
            for y in range(pad_size, image.shape[1] + pad_size):
                window = image_padded[x-pad_size:x+pad_size+1, y-pad_size:y+pad_size+1].flatten()
                if blur_type == 'mean':
                    result[x-pad_size, y-pad_size] = np.mean(window)
                elif blur_type == 'median':
                    result[x-pad_size, y-pad_size] = np.median(window)
                elif blur_type == 'mode':
                    result[x-pad_size, y-pad_size] = np.argmax(np.bincount(window))
        return result

    def blur_images(self, images, kernel_size=3, blur_type='mean'):
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(executor.map(self.blur_single, images, repeat(kernel_size), repeat(blur_type)))
        return np.array(results)

class EdgeDetection:
    def __init__(self, kernels=None, num_workers=1):
        self.num_workers = num_workers
        self.kernels = kernels if kernels else {
            'sobel': (np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]), np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])),
            'prewitt': (np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]), np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])),
            'robert': (np.array([[1, 0], [0, -1]]), np.array([[0, 1], [-1, 0]]))
        }

    def combine(self, vertical, horizontal):
        return np.clip(np.abs(vertical) + np.abs(horizontal), 0, 255)

    def convolution(self, image, kernel):
        pad_size = (kernel.shape[0] // 2, kernel.shape[1] // 2)
        image_padded = np.pad(image, pad_size, mode="constant")
        result = np.zeros_like(image)
        for x in range(image.shape[0]):
            for y in range(image.shape[1]):
                result[x, y] = np.sum(kernel * image_padded[x:x+kernel.shape[0], y:y+kernel.shape[1]])
        return result

    def edge_detection_single(self, args):
        image, method = args
        kernel_x, kernel_y = self.kernels[method]
        vertical = self.convolution(image, kernel_x)
        horizontal = self.convolution(image, kernel_y)
        return self.combine(vertical, horizontal)

    def edge_detection(self, images, method):
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(executor.map(self.edge_detection_single, [(image, method) for image in images]))
        return np.array(results)
class Morphology:
    def __init__(self, kernel= np.ones((3,3), np.uint8), num_workers=1):
        self.kernel = kernel
        self.num_workers = num_workers

    def threshold(self, image, binary_threshold):
        return np.where(image >= binary_threshold, 1, 0)

    def binary(self, images, binary_threshold=128):
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(executor.map(self.threshold, images, repeat(binary_threshold)))
        return np.array(results)
    
    def apply_operation(self, images, operation):
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(executor.map(operation, images))
        return results
    
    def dilation_single(self, image):
        matrix = np.array(image)
        result = np.zeros(matrix.shape)
        matrixPad = np.pad(matrix, (self.kernel.shape[0]//2, self.kernel.shape[1]//2), mode="constant")
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                if np.any(self.kernel == matrixPad[i:i+self.kernel.shape[0], j:j+self.kernel.shape[1]]):
                    result[i, j] = 1
        return result
        
    def dilation(self, images):
        return self.apply_operation(images, self.dilation_single)
    
    def erosion_single(self, image):
        matrix = np.array(image)
        result = np.zeros(matrix.shape)
        matrixPad = np.pad(matrix, (self.kernel.shape[0]//2, self.kernel.shape[1]//2), mode="constant")
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                if np.all(self.kernel == matrixPad[i:i+self.kernel.shape[0], j:j+self.kernel.shape[1]]):
                    result[i, j] = 1
        return result  
    
    def erosion(self, images):
        return self.apply_operation(images, self.erosion_single)
    
    def opening(self, images):
        return self.dilation(self.erosion(images))
    
    def closing(self, images):
        return self.erosion(self.dilation(images))
    
    def gradient_single(args):
        dilasi_img, erosi_img = args
        result = np.zeros(dilasi_img.shape)
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                result[i,j] = max(0, dilasi_img[i,j] - erosi_img[i,j])
        return result

    def gradient(self, images):
        dilated = self.dilation(images)
        eroded = self.erosion(images)
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(executor.map(self.gradient_single, zip(dilated, eroded)))
        return results


    
