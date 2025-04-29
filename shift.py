# //////////////////////////////////
import cv2
import numpy as np
import matplotlib.pyplot as plt




class MeanShiftSegmentation:
    def __init__(self, image):
        self.image = image
        self.height, self.width = image.shape[:2]


        
    def segment_image(self, color_threshold=5, spatial_threshold=0, max_iterations=20):
        """Segments the image using mean shift with colorful output"""
        # Convert to feature space (color + position)
        features = self._create_feature_space()
        
        # Initialize output with random colors
        output = np.zeros_like(self.image)
        remaining_pixels = features.copy()
        np.random.shuffle(remaining_pixels)  # Random processing order
        
        while len(remaining_pixels) > 0:
            # Pick random seed point
            current_point = remaining_pixels[0]
            
            # Mean shift iterations
            for _ in range(max_iterations):
                # Find neighbors
                distances = self._calculate_distances(remaining_pixels, current_point, 
                                                     color_threshold, spatial_threshold)
                neighbors = remaining_pixels[distances]
                
                # Update mean
                new_mean = np.mean(neighbors, axis=0)
                if np.linalg.norm(new_mean - current_point) < 1:  # Convergence check
                    break
                current_point = new_mean
            
            # Assign random color to cluster
            cluster_color = np.random.randint(0, 256, 3, dtype=np.uint8)
            self._color_cluster(output, remaining_pixels, current_point, 
                               color_threshold, spatial_threshold, cluster_color)
            
            # Remove processed pixels
            remaining_pixels = remaining_pixels[~distances]
            
        return output

    def _create_feature_space(self):
        """Create feature vectors with normalized color and position"""
        positions = np.indices((self.height, self.width)).transpose(1,2,0)
        features = np.concatenate((
            self.image.astype(np.float32),  # Color features
            positions.astype(np.float32)     # Spatial features
        ), axis=-1)
        return features.reshape(-1, 5)

    def _calculate_distances(self, pixels, center, c_thresh, s_thresh):
        """Calculate combined color and spatial distances"""
        color_dist = np.linalg.norm(pixels[:, :3] - center[:3], axis=1)
        spatial_dist = np.linalg.norm(pixels[:, 3:] - center[3:], axis=1)
        return (color_dist < c_thresh) & (spatial_dist < s_thresh)

    def _color_cluster(self, output, pixels, center, c_thresh, s_thresh, color):
        """Color all pixels in the cluster"""
        mask = self._calculate_distances(pixels, center, c_thresh, s_thresh)
        y_coords = pixels[mask, 3].astype(int)
        x_coords = pixels[mask, 4].astype(int)
        output[y_coords, x_coords] = color

# def visualize_results(image_path):
#     # Load image
#     original = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    
#     # Process image
#     segmenter = MeanShiftSegmentation(original)
#     segmented = segmenter.segment_image(
#         color_threshold=60, 
#         spatial_threshold=250,
#         max_iterations=15
#     )
    
#     # Create visualization
#     plt.figure(figsize=(15, 8))
    
#     plt.subplot(1, 2, 1)
#     plt.imshow(original)
#     plt.title('Original Image')
#     plt.axis('off')
    
#     plt.subplot(1, 2, 2)
#     plt.imshow(segmented)
#     plt.title('Colorful Mean Shift Segmentation')
#     plt.axis('off')
    
#     plt.tight_layout()
#     plt.show()

# # Usage - try with different image sizes
# visualize_results('small.jpg')