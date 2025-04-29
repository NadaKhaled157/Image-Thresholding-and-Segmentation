# # //////////////////////////////////
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt




# class MeanShiftSegmentation:
#     def __init__(self, image):
#         if len(image.shape) == 2:
#             self.image = np.stack([image]*3, axis=-1)
#         else:
#             self.image = image
#         self.height, self.width = self.image.shape[:2]


#     def segment_image(self, color_threshold=5, spatial_threshold=0, max_iterations=20):
#         """Segments the image using mean shift with colorful output"""
#         # Convert to feature space (color + position)
#         features = self._create_feature_space()
        
#         # Initialize output with random colors
#         output = np.zeros_like(self.image)
#         remaining_pixels = features.copy()
#         np.random.shuffle(remaining_pixels)  # Random processing order
        
#         while len(remaining_pixels) > 0:
#             # Pick random seed point
#             current_point = remaining_pixels[0]
            
#             # Mean shift iterations
#             for _ in range(max_iterations):
#                 # Find neighbors
#                 distances = self._calculate_distances(remaining_pixels, current_point, 
#                                                      color_threshold, spatial_threshold)
#                 neighbors = remaining_pixels[distances]
                
#                 # Update mean
#                 new_mean = np.mean(neighbors, axis=0)
#                 if np.linalg.norm(new_mean - current_point) < 1:  # Convergence check
#                     break
#                 current_point = new_mean
            
#             # Assign random color to cluster
#             cluster_color = np.random.randint(0, 256, 3, dtype=np.uint8)
#             self._color_cluster(output, remaining_pixels, current_point, 
#                                color_threshold, spatial_threshold, cluster_color)
            
#             # Remove processed pixels
#             remaining_pixels = remaining_pixels[~distances]
            
#         return output

#     def _create_feature_space(self):
#         """Create feature vectors with normalized color and position"""
#         positions = np.indices((self.height, self.width)).transpose(1,2,0)
#         features = np.concatenate((
#             self.image.astype(np.float32),  # Color features
#             positions.astype(np.float32)     # Spatial features
#         ), axis=-1)
#         return features.reshape(-1, 5)

#     def _calculate_distances(self, pixels, center, c_thresh, s_thresh):
#         """Calculate combined color and spatial distances"""
#         color_dist = np.linalg.norm(pixels[:, :3] - center[:3], axis=1)
#         spatial_dist = np.linalg.norm(pixels[:, 3:] - center[3:], axis=1)
#         return (color_dist < c_thresh) & (spatial_dist < s_thresh)

#     def _color_cluster(self, output, pixels, center, c_thresh, s_thresh, color):
#         """Color all pixels in the cluster"""
#         mask = self._calculate_distances(pixels, center, c_thresh, s_thresh)
#         y_coords = pixels[mask, 3].astype(int)
#         x_coords = pixels[mask, 4].astype(int)
#         output[y_coords, x_coords] = color



################################################################################################\



import cv2
import numpy as np
import matplotlib.pyplot as plt

class MeanShiftSegmentation:
    def __init__(self, image):
        if len(image.shape) == 2:
            self.image = np.stack([image]*3, axis=-1)
        else:
            self.image = image
        self.height, self.width = self.image.shape[:2]
        
    def segment_image(self, color_threshold=5, spatial_threshold=0, max_iterations=20):
  
        features = self._create_feature_space()
        output = np.zeros_like(self.image)
        remaining_pixels = features.copy()
        np.random.shuffle(remaining_pixels)
        
        while len(remaining_pixels) > 0:
            current_point = remaining_pixels[0]
            
            # Mean shift iterations
            for _ in range(max_iterations):
                distances = self._calculate_distances(remaining_pixels, current_point, 
                                                     color_threshold, spatial_threshold)
                neighbors = remaining_pixels[distances]
                new_mean = np.mean(neighbors, axis=0)
                if np.linalg.norm(new_mean - current_point) < 1:
                    break
                current_point = new_mean
            
            # Calculate mean color from cluster
            mean_color = current_point[:3].astype(np.uint8)
            self._color_cluster(output, remaining_pixels, current_point, 
                               color_threshold, spatial_threshold, mean_color)
            
            remaining_pixels = remaining_pixels[~distances]


            
        return output

    def _create_feature_space(self):
      
        positions = np.indices((self.height, self.width)).transpose(1,2,0)
        features = np.concatenate((
            self.image.astype(np.float32),
            positions.astype(np.float32)
        ), axis=-1)
        return features.reshape(-1, 5)

    def _calculate_distances(self, pixels, center, c_thresh, s_thresh):
     
        color_dist = np.linalg.norm(pixels[:, :3] - center[:3], axis=1)
        spatial_dist = np.linalg.norm(pixels[:, 3:] - center[3:], axis=1)
        return (color_dist < c_thresh) & (spatial_dist < s_thresh)

    def _color_cluster(self, output, pixels, center, c_thresh, s_thresh, color):
       
        mask = self._calculate_distances(pixels, center, c_thresh, s_thresh)
        y_coords = pixels[mask, 3].astype(int)
        x_coords = pixels[mask, 4].astype(int)
        output[y_coords, x_coords] = color

