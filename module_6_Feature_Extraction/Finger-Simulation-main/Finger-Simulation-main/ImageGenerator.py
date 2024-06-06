import numpy as np
import math
import random
import os
from matplotlib import pyplot as plt
import cv2
from scipy.ndimage import distance_transform_edt
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
class ImageAugmentation:
    def __init__(self, rotation_range=0, width_shift_range=0, height_shift_range=0, zoom_range=0, horizontal_flip=False, vertical_flip=False):
        self.rotation_range = rotation_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.zoom_range = zoom_range
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip

    def shuffle(self, images, labels):
        indices = list(range(len(images)))
        random.shuffle(indices)
        images = [images[i] for i in indices]
        labels = [labels[i] for i in indices]
        return images, labels
    
    def random_transform(self, image):
        if self.zoom_range:
            zoom = random.uniform(1 - self.zoom_range, 1 + self.zoom_range)
            image = self.zoom(image, zoom)
            image
        if self.rotation_range:
            angle = random.uniform(-self.rotation_range, self.rotation_range)
            image = self.rotate(image, angle)
        if self.width_shift_range:
            shift = random.uniform(-self.width_shift_range, self.width_shift_range)
            image = self.translate(image, int(shift), 0)
        if self.height_shift_range:
            shift = random.uniform(-self.height_shift_range, self.height_shift_range)
            image = self.translate(image, 0, int(shift))
        if self.horizontal_flip and random.random() < 0.5:
            image = self.flip(image, 1)
        if self.vertical_flip and random.random() < 0.5:
            image = self.flip(image, 0)
        return image

    def flow(self, images, labels, num_augmentations=1, shuffle=False):
        augmented_images = []
        augmented_labels = []
        max_workers = os.cpu_count() // 2

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for image, label in zip(images, labels):
                augmented_images.append(image)
                augmented_labels.append(label)
                transformations = list(executor.map(self.random_transform, [image]*num_augmentations))
                augmented_images.extend(transformations)
                augmented_labels.extend([label] * num_augmentations)

        if shuffle:
            augmented_images, augmented_labels = self.shuffle(augmented_images, augmented_labels)
        print(f'Generated {len(augmented_images)} augmented images.')
        augmented_images = np.array(augmented_images)
        augmented_labels = np.array(augmented_labels)
        return augmented_images, augmented_labels

    def random_transform(self, image):
        if self.zoom_range:
            zoom = random.uniform(1 - self.zoom_range, 1 + self.zoom_range)
            image = self.zoom(image, zoom)
            image
        if self.rotation_range:
            angle = random.uniform(-self.rotation_range, self.rotation_range)
            image = self.rotate(image, angle)
        if self.width_shift_range:
            shift = random.uniform(-self.width_shift_range, self.width_shift_range)
            image = self.translate(image, int(shift), 0)
        if self.height_shift_range:
            shift = random.uniform(-self.height_shift_range, self.height_shift_range)
            image = self.translate(image, 0, int(shift))
        if self.horizontal_flip and random.random() < 0.5:
            image = self.flip(image, 1)
        if self.vertical_flip and random.random() < 0.5:
            image = self.flip(image, 0)
        return image
    
    def rotate(self, image, angle):
        angle = math.radians(angle)
        height, width = image.shape[:2]
        y, x = np.indices((height, width))
        y_new = (y - height / 2) * math.cos(angle) - (x - width / 2) * math.sin(angle) + height / 2
        x_new = (y - height / 2) * math.sin(angle) + (x - width / 2) * math.cos(angle) + width / 2
        x_new, y_new = x_new.astype(int), y_new.astype(int)
        mask = (0 <= x_new) & (x_new < width) & (0 <= y_new) & (y_new < height)
        image_rotated = np.zeros_like(image)
        image_rotated[y[mask], x[mask]] = image[y_new[mask], x_new[mask]]
        return self.fill_nearest(image_rotated)

    def flip(self, image, axis):
        if axis == 0:
            return image[::-1, :]
        elif axis == 1:
            return image[:, ::-1]

    def translate(self, image, x, y):
        height, width = image.shape[:2]
        y_new, x_new = np.indices((height, width))
        y_new, x_new = y_new - y, x_new - x
        mask = (0 <= x_new) & (x_new < width) & (0 <= y_new) & (y_new < height)
        image_translated = np.zeros_like(image)
        image_translated[np.where(mask)] = image[y_new[mask], x_new[mask]]
        return self.fill_nearest(image_translated)
    
    def zoom(self, image, zoom):
        height, width = image.shape[:2]
        y, x = np.indices((height, width))
        y_new = (y - height / 2) / zoom + height / 2
        x_new = (x - width / 2) / zoom + width / 2
        x_new, y_new = x_new.astype(int), y_new.astype(int)
        mask = (0 <= x_new) & (x_new < width) & (0 <= y_new) & (y_new < height)
        image_zoomed = np.zeros_like(image)
        image_zoomed[y[mask], x[mask]] = image[y_new[mask], x_new[mask]]
        return self.fill_nearest(image_zoomed)

    def fill_nearest(self, image):
        mask = np.all(image == 0, axis=-1)
        distances, indices = distance_transform_edt(mask, return_indices=True)
        nearest_pixels = image[tuple(indices)]
        image[mask] = nearest_pixels[mask]
        return image
    
class ImageDataGenerator:
    def __init__(self, rotation_range=0, width_shift_range=0, height_shift_range=0, zoom_range=0, horizontal_flip=False, vertical_flip=False):
        self.augmentor = ImageAugmentation(rotation_range, width_shift_range, height_shift_range, zoom_range, horizontal_flip, vertical_flip)

    def resize(self, image, size):
        height, width = image.shape[:2]
        new_height, new_width = size
        y_new, x_new = np.indices((new_height, new_width))
        y_new, x_new = (y_new * height // new_height).astype(int), (x_new * width // new_width).astype(int)
        image_resized = image[y_new, x_new]
        return image_resized
    
    def shuffle(self, images, labels):
        indices = list(range(len(images)))
        np.random.shuffle(indices)
        images = [images[i] for i in indices]
        labels = [labels[i] for i in indices]
        return images, labels

    def process_image(self, image_path, target_size):
        image = plt.imread(image_path)
        image = self.resize(image, target_size)
        image = self.augmentor.random_transform(image)
        return image

    def flow_from_directory(self, directory, target_size, shuffle=False):
        images = []
        labels = []
        max_workers = os.cpu_count() // 2

    def process_path(self, path, target_size):
        return self.process_image(path, target_size)

    def flow_from_directory(self, directory, target_size, shuffle=False):
        images = []
        labels = []
        max_workers = os.cpu_count() // 2

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for subfolder in os.listdir(directory):
                subfolder_path = os.path.join(directory, subfolder)
                if os.path.isdir(subfolder_path):
                    image_paths = [os.path.join(subfolder_path, filename) for filename in os.listdir(subfolder_path)]
                    image_results = list(executor.map(self.process_path, image_paths, repeat(target_size)))
                    images.extend(image_results)
                    labels.extend([subfolder] * len(image_results))

        if shuffle:
            images, labels = self.shuffle(images, labels)

        print(f'Found {len(images)} images belonging to {len(set(labels))} classes.')
        images = np.array(images)
        labels = np.array(labels)
        return images, labels
    