import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def open_to_rgb(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image_rgb

def segment(image_rgb, K):

    # Reshape and cast to float
    pixel_values = image_rgb.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    # Generate clustering object and apply to image data
    kmeans = KMeans(n_clusters=K, random_state=0)
    kmeans.fit(pixel_values)

    # Fetch centroids and map pixel values to closest centroid
    segmented_labels = kmeans.labels_
    centroids = np.uint8(kmeans.cluster_centers_)

    segmented_image = centroids[segmented_labels.flatten()]
    segmented_image = segmented_image.reshape(image_rgb.shape)

    return segmented_image

image_rgb = open_to_rgb('cats2.png')
clusters = [2, 5, 10]

plt.figure(figsize=(12, 6))

# Plot image for every cluster number
for i, K in enumerate(clusters, start=1):
    segmented_image = segment(image_rgb, K)
    plt.subplot(2, 2, i)
    plt.title(f'K = {K}')
    plt.imshow(segmented_image)
    plt.axis('off')

# Plot original image
plt.subplot(2, 2, 4)
plt.title('Original Image')
plt.imshow(image_rgb)
plt.axis('off')

plt.show()