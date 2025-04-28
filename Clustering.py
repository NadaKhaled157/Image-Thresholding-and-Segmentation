import numpy as np


def kmeans_image_clustering(image, k=3, max_iters=10):
    """
    Apply simple KMeans clustering on an image and visualize the clusters.
    
    Parameters:
    - image: input image (numpy array)
    - is_rgb: True if image is RGB, False if grayscale
    - k: number of clusters
    - max_iters: maximum number of iterations
    
    Returns:
    - clustered_image: the image with cluster colors
    """
    if len(image.shape) == 2:
     is_rgb = False
    else:
        is_rgb = True



    # 1. Prepare the data
    if is_rgb:
        # For RGB images: each pixel has 3 values (R, G, B)
        data = image.reshape(-1, 3).astype(np.float32)
    else:
        # For grayscale images: each pixel has 1 value
        data = image.reshape(-1, 1).astype(np.float32)
    
    # 2. Randomly initialize centroids
    np.random.seed(0)
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]

    # 3. Iteratively update centroids
    for iteration in range(max_iters):
        # 3.1 Compute distances between each pixel and each centroid
        distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
        
        # 3.2 Assign each pixel to the nearest centroid
        labels = np.argmin(distances, axis=1)
        
        # 3.3 Calculate new centroids
        new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])
        
        # 3.4 Update centroids
        centroids = new_centroids

    # 4. Create the clustered image
    clustered_data = centroids[labels].reshape(image.shape)

    # 5. Convert to appropriate type
    clustered_image = np.clip(clustered_data, 0, 255).astype(np.uint8)

    return clustered_image

