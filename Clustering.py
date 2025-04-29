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





def region_growing(img, seed_point, threshold=20):
    """
    img: Input image (grayscale or colored)
    seed_point: Starting point (x, y)
    threshold: Maximum allowed color difference
    """
    if len(img.shape) == 2:
        is_color = False  # Grayscale
    else:
        is_color = True   # Colored (RGB or BGR)

    height, width = img.shape[:2]
    visited = np.zeros((height, width), dtype=np.bool_)
    region = np.zeros((height, width), dtype=np.uint8)

    if is_color:
        seed_value = img[seed_point[1], seed_point[0], :].astype(np.float32)
    else:
        seed_value = float(img[seed_point[1], seed_point[0]])

    queue = [seed_point]
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1),
                 (-1, -1), (-1, 1), (1, -1), (1, 1)]

    count = 1

    while queue:
        x, y = queue.pop(0)
        visited[y, x] = True
        region[y, x] = 255  # Mark the region in white

        for dx, dy in neighbors:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height:
                if not visited[ny, nx]:
                    if is_color:
                        neighbor_value = img[ny, nx, :].astype(np.float32)
                        error = np.linalg.norm(neighbor_value - seed_value)
                    else:
                        neighbor_value = float(img[ny, nx])
                        error = abs(neighbor_value - seed_value)

                    if error <= threshold:
                        queue.append((nx, ny))
                        visited[ny, nx] = True
                        if is_color:
                            seed_value = (seed_value * count + neighbor_value) / (count + 1)
                        else:
                            seed_value = (seed_value * count + neighbor_value) / (count + 1)
                        count += 1
                    else:
                        visited[ny, nx] = True  # Mark as visited without adding

    return region




######################################################## Agglomerative ########################################################

def initial_clusters(image_clusters, k):
    """
    Initializes the clusters with k colors by grouping the image pixels based on their color.
    """
    # Determine the range of colors for each cluster
    cluster_color = int(256 / k)
    # Initialize empty groups for each cluster
    groups = [[] for _ in range(k)]

    # Assign each pixel to its closest cluster based on color
    for p in image_clusters:
        # Calculate the mean color of the pixel
        color = int(np.mean(p))
        # Determine the index of the closest cluster
        group_index = min(range(k), key=lambda i: abs(
            color - i * cluster_color))
        # Add the pixel to the corresponding cluster
        groups[group_index].append(p)

    # Remove empty clusters if any
    return [group for group in groups if group]

def get_cluster_center(cluster):
    """
    Returns the center of the cluster.
    """
    # Calculate the mean of all points in the cluster
    return np.mean(cluster, axis=0)

def get_clusters(image_clusters, clusters_number):
    """
    Agglomerative clustering algorithm to group the image pixels into a specified number of clusters.
    """
    # Initialize clusters and their assignments
    clusters_list = initial_clusters(image_clusters, clusters_number)
    cluster_assignments = {tuple(point): i for i, cluster in enumerate(
        clusters_list) for point in cluster}
    # Calculate initial cluster centers
    centers = [get_cluster_center(cluster)
                for cluster in clusters_list]

    # Merge clusters until the desired number is reached
    while len(clusters_list) > clusters_number:
        min_distance = float('inf')
        merge_indices = None

        # Find the two clusters with the minimum distance
        for i, cluster1 in enumerate(clusters_list):
            for j, cluster2 in enumerate(clusters_list[:i]):
                distance = get_euclidean_distance(
                    centers[i], centers[j])
                if distance < min_distance:
                    min_distance = distance
                    merge_indices = (i, j)

        # Merge the closest clusters
        i, j = merge_indices
        clusters_list[i] += clusters_list[j]
        del clusters_list[j]
        # Update cluster centers
        centers[i] = get_cluster_center(clusters_list[i])
        del centers[j]

        # Update cluster assignments
        for point in clusters_list[i]:
            cluster_assignments[tuple(point)] = i

    return cluster_assignments, centers



def get_euclidean_distance( pixel, centroid):
        return np.linalg.norm(pixel - centroid)

def apply_agglomerative_clustering(image, clusters_number):
    """
    Applies agglomerative clustering to the image (RGB or Grayscale) and returns the segmented image.
    """
    # Check if the image is grayscale or RGB
    if len(image.shape) == 2:  # Grayscale image
        flattened_image = image.reshape((-1, 1))
    else:  # RGB image
        flattened_image = image.reshape((-1, 3))
    
    # Perform agglomerative clustering
    cluster_assignments, centers = get_clusters(flattened_image, clusters_number)
    
    # Assign each pixel in the image to its corresponding cluster center
    output_image = np.array([centers[cluster_assignments[tuple(p)]] for p in flattened_image], dtype=np.uint8)
    
    # Reshape the segmented image to its original shape
    if len(image.shape) == 2:
        output_image = output_image.reshape(image.shape)
    else:
        output_image = output_image.reshape(image.shape)
    return output_image

