import sys
import cv2
import numpy as np
import trimesh
from PIL import Image
from rembg import remove
import io
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                           QPushButton, QLabel, QFileDialog, QMessageBox, QGroupBox, QComboBox,
                           QSlider, QDoubleSpinBox, QFormLayout, QCheckBox, QProgressBar)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt, QThread, pyqtSignal
import pyqtgraph.opengl as gl

class RembgBackgroundRemover:
    
    def __init__(self):
        pass
    
    def remove_background(self, image):
        
        # Convert OpenCV image (BGR) to PIL Image (RGB)
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        # Use rembg to remove background
        output = remove(pil_img)
        
        # Convert PIL image back to numpy array (with alpha channel)
        output_array = np.array(output)
        
        # Convert from RGBA to BGRA
        bgra = cv2.cvtColor(output_array, cv2.COLOR_RGBA2BGRA)
        
        # Create binary mask from alpha channel
        mask = (bgra[:, :, 3] > 0).astype(np.uint8) * 255
        
        return bgra, mask

class EnhancedMeshGenerator(QThread):
    finished = pyqtSignal(np.ndarray, np.ndarray, np.ndarray)
    mesh_ready = pyqtSignal(trimesh.Trimesh)
    progress = pyqtSignal(int)

    def __init__(self, image, depth_strength=1.0, extrusion_depth=0.5, add_base=True, invert_depth=True, real_dimensions=None):
        super().__init__()
        self.image = image
        self.depth_strength = depth_strength
        self.extrusion_depth = extrusion_depth
        self.add_base = add_base
        self.invert_depth = invert_depth  # New parameter to control depth inversion
        self.real_dimensions = real_dimensions  # (width, height, depth) in mm

    def estimate_depth(self, image, contour_mask, invert_depth=True):
        """Advanced depth estimation with option for inverted or regular depth"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Ensure proper data types
        gray = gray.astype(np.uint8)
        contour_mask = contour_mask.astype(np.uint8)
        
        # Edge detection with multiple thresholds
        edges1 = cv2.Canny(gray, 30, 100)
        edges2 = cv2.Canny(gray, 50, 150)
        
        # Combine edges with weighted blending
        edges = cv2.addWeighted(edges1, 0.4, edges2, 0.6, 0)
        
        # Calculate structure tensor for depth cues
        sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        
        structure_tensor = np.sqrt(sobelx**2 + sobely**2)
        structure_tensor = cv2.normalize(structure_tensor, None, 0, 1, cv2.NORM_MINMAX)
        
        # Apply distance transform (low weight)
        dist_transform = cv2.distanceTransform(contour_mask, cv2.DIST_L2, 3)
        dist_transform = cv2.normalize(dist_transform, None, 0, 1.0, cv2.NORM_MINMAX)
        
        # Intensity-based depth - this will be affected by the inversion option
        intensity_depth = gray.astype(np.float32) / 255.0
        
        # Apply inversion based on user choice
        if invert_depth:
            # Invert to make dark areas pop up (higher)
            intensity_depth = 1.0 - intensity_depth
        # In non-inverted mode, bright areas will be higher
        
        intensity_depth = cv2.normalize(intensity_depth, None, 0, 1.0, cv2.NORM_MINMAX)
        
        # Enhanced texture detection
        # This captures local variations in the image texture
        texture_kernel = np.ones((5, 5), np.float32) / 25
        local_mean = cv2.filter2D(gray.astype(np.float32), -1, texture_kernel)
        texture_detail = np.abs(gray.astype(np.float32) - local_mean)
        texture_detail = cv2.normalize(texture_detail, None, 0, 1, cv2.NORM_MINMAX)
        
        # Apply Laplacian for fine detail enhancement
        # This helps bring out fine details in the image
        laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=1)
        laplacian = np.abs(laplacian)
        laplacian = cv2.normalize(laplacian, None, 0, 1, cv2.NORM_MINMAX)
        
        # Extract internal details using multiple approaches
        # 1. Local contrast variation (enhances texture details)
        local_contrast = cv2.bilateralFilter(gray, 9, 75, 75).astype(np.float32)
        local_contrast = np.abs(local_contrast - gray.astype(np.float32)) / 255.0
        local_contrast = cv2.normalize(local_contrast, None, 0, 1, cv2.NORM_MINMAX)
        
        # 2. High-pass filter to extract fine details
        blurred = cv2.GaussianBlur(gray.astype(np.float32), (15, 15), 0)
        high_pass = np.abs(gray.astype(np.float32) - blurred) / 255.0
        high_pass = cv2.normalize(high_pass, None, 0, 1, cv2.NORM_MINMAX)
        
        # 3. Use color information to capture detail that might be lost in grayscale
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hue, saturation, value = cv2.split(hsv)
        saturation = saturation.astype(np.float32) / 255.0
        
        # Combine all depth cues with weights that emphasize internal details
        depth_map = (
            intensity_depth * 0.35 +  # Major contributor - makes dark/light areas pop up based on inversion
            texture_detail * 0.20 +   # Enhances texture variations
            high_pass * 0.15 +        # Emphasizes fine details
            local_contrast * 0.15 +   # Adds local contrast variations
            laplacian * 0.10 +        # Adds edge detail
            saturation * 0.05         # Uses color information
        )
        
        # Subtract edge influence to make the outlines less prominent
        edge_influence = cv2.dilate(edges.astype(np.float32) / 255.0, np.ones((3, 3), np.uint8))
        edge_influence = cv2.GaussianBlur(edge_influence, (5, 5), 0)
        edge_influence = cv2.normalize(edge_influence, None, 0, 0.3, cv2.NORM_MINMAX)  # Reduced strength
        
        # Subtract edge influence from the depth map (lowers the borders)
        depth_map = cv2.normalize(depth_map - edge_influence * 0.5, None, 0, 1, cv2.NORM_MINMAX)
        
        # Apply smoothing while preserving details
        try:
            # Create detail layer
            detail_layer = depth_map - cv2.GaussianBlur(depth_map, (7, 7), 0)
            detail_layer = cv2.normalize(detail_layer, None, 0, 0.2, cv2.NORM_MINMAX)
            
            # Apply bilateral filter to preserve edges
            smoothed = cv2.bilateralFilter(depth_map, 7, 50, 50)
            
            # Combine smoothed base with enhanced details
            depth_map = smoothed + detail_layer
            depth_map = cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX)
        except Exception as e:
            print(f"Filtering error (non-critical): {str(e)}")
            # Fallback to simple smoothing
            depth_map = cv2.GaussianBlur(depth_map, (3, 3), 0)
        
        # Apply adaptive histogram equalization for better contrast
        try:
            depth_uint8 = (depth_map * 255).astype(np.uint8)
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            depth_map = clahe.apply(depth_uint8).astype(np.float32) / 255
        except Exception as e:
            print(f"CLAHE error (non-critical): {str(e)}")
        
        # Remove mountain-like features but preserve intentional height variations
        try:
            depth_map = self.detect_and_remove_mountains(depth_map, contour_mask)
        except Exception as e:
            print(f"Mountain removal error (non-critical): {str(e)}")
        
        # Apply the depth strength multiplier
        depth_map = depth_map * self.depth_strength
        
        return depth_map

    def detect_and_remove_mountains(self, depth_map, shape_mask):
       
        # Ensure proper data types to prevent OpenCV errors
        depth_map = depth_map.astype(np.float32)
        shape_mask = shape_mask.astype(np.uint8)
        
        # Only process areas in the shape mask
        masked_depth = depth_map * (shape_mask > 0).astype(np.float32)
        
        # Calculate local max values using dilation
        kernel_size = 11  # Adjust based on the size of mountains to detect
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        # Normalize depth for dilation (prevent OpenCV errors)
        depth_norm = (masked_depth * 255).astype(np.uint8)
        dilated = cv2.dilate(depth_norm, kernel)
        
        # Find local maxima (where dilated == original)
        local_max = (np.abs(dilated - depth_norm) < 3) & (depth_norm > 0)
        local_max = local_max.astype(np.uint8) * 255
        
        # Calculate gradients with proper formats
        # Convert to 8-bit for gradient calculation to avoid OpenCV errors
        depth_8bit = (masked_depth * 255).astype(np.uint8)
        gradient_x = cv2.Sobel(depth_8bit, cv2.CV_32F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(depth_8bit, cv2.CV_32F, 0, 1, ksize=3)
        
        # Calculate gradient magnitude
        gradient_mag = np.sqrt(gradient_x**2 + gradient_y**2)
        
        # Find areas with large gradients
        # Use a fixed threshold instead of percentile to avoid potential errors
        gradient_threshold = 100.0  # Adjust as needed
        steep_areas = gradient_mag > gradient_threshold
        
        # Convert to proper format
        steep_areas = steep_areas.astype(np.uint8)
        
        # Identify potential mountain regions
        # A mountain is a local maximum with steep areas nearby
        kernel = np.ones((kernel_size*2, kernel_size*2), np.uint8)
        steep_areas_dilated = cv2.dilate(steep_areas, kernel)
        
        # Mountains are local maxima with steep areas nearby
        mountain_mask = (local_max > 0) & (steep_areas_dilated > 0)
        
        # Optionally, cluster nearby mountain points
        mountain_mask = mountain_mask.astype(np.uint8) * 255
        mountain_mask = cv2.dilate(mountain_mask, np.ones((5, 5), np.uint8))
        
        # For each detected mountain, flatten the area
        if np.any(mountain_mask):
            try:
                # Find connected components (mountain regions)
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mountain_mask)
                
                # Process each mountain region
                for i in range(1, num_labels):  # Skip label 0 (background)
                    # Get the region of this mountain
                    mountain_region = (labels == i)
                    
                    # Find the boundary of the mountain region
                    kernel = np.ones((3, 3), np.uint8)
                    eroded = cv2.erode(mountain_region.astype(np.uint8), kernel)
                    boundary = mountain_region & ~(eroded.astype(bool))
                    
                    if np.any(boundary):
                        # Get the average height at the boundary
                        boundary_heights = masked_depth[boundary]
                        if len(boundary_heights) > 0:
                            avg_boundary_height = np.mean(boundary_heights)
                            
                            # Flatten the mountain region to this height
                            # This effectively removes the peak while maintaining the base
                            masked_depth[mountain_region] = avg_boundary_height
            except Exception as e:
                print(f"Mountain detection error (non-critical): {str(e)}")
                # Continue with the original depth map if there's an error
                pass
        
        # Apply a gaussian blur to smooth the modified areas
        masked_depth = cv2.GaussianBlur(masked_depth, (5, 5), 0)
        
        return masked_depth

    def detect_shape(self, image):
        """Enhanced shape detection optimized for images with alpha channel"""
        # Special handling for images with alpha channel
        if image.shape[2] == 4:
            # Extract alpha channel
            alpha = image[:, :, 3]
            
            # Multi-threshold approach for better edge definition
            # 1. Strong threshold for definite object parts
            _, mask_strong = cv2.threshold(alpha, 200, 255, cv2.THRESH_BINARY)
            
            # 2. Medium threshold for semi-transparent areas that should be included
            _, mask_medium = cv2.threshold(alpha, 30, 255, cv2.THRESH_BINARY)
            
            # 3. Adaptive threshold to handle varying transparency levels
            adaptive_thresh = cv2.adaptiveThreshold(
                alpha, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, -2  # Negative constant to be more inclusive
            )
            
            # Combine masks with priority to the stronger signals
            # Start with adaptive for maximum coverage
            combined_mask = adaptive_thresh.copy()
            # Medium threshold overrides where it's set
            combined_mask = cv2.bitwise_or(combined_mask, mask_medium)
            # Strong threshold has final say
            combined_mask = cv2.bitwise_or(combined_mask, mask_strong)
            
            # Clean up the mask with morphological operations
            kernel_small = np.ones((3, 3), np.uint8)
            kernel_medium = np.ones((5, 5), np.uint8)
            
            # Close small holes first (connect nearby components)
            mask_closed = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_small, iterations=2)
            
            # Fill any remaining internal holes using contour filling
            contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            mask_filled = np.zeros_like(mask_closed)
            
            if contours:
                # Keep only significant contours (filter out tiny noise)
                min_area = (alpha.shape[0] * alpha.shape[1]) * 0.001  # 0.1% of image area
                valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
                
                # Draw all valid contours filled
                cv2.drawContours(mask_filled, valid_contours, -1, 255, -1)
                
                # Optional: Perform one more closing to ensure smooth boundaries
                mask_filled = cv2.morphologyEx(mask_filled, cv2.MORPH_CLOSE, kernel_medium)
                
                # Dilate slightly to ensure we capture all of the object's edges
                mask_filled = cv2.dilate(mask_filled, kernel_small, iterations=1)
                
                return mask_filled
            
            # If no valid contours found, return the closed mask
            return mask_closed
        
        # For non-alpha images, use the existing shape detection method
        else:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply advanced noise reduction
            denoised = cv2.fastNlMeansDenoising(gray, None, 20, 7, 21)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(denoised, (5, 5), 0)
            
            # Try multiple thresholding approaches and combine results
            # 1. Adaptive thresholding
            adaptive_thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 11, 2
            )
            
            # 2. Otsu's thresholding
            _, otsu_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # 3. Edge-based segmentation
            edges = cv2.Canny(blurred, 30, 200)
            kernel = np.ones((5, 5), np.uint8)
            edges_dilated = cv2.dilate(edges, kernel, iterations=1)
            
            # Combine thresholding results
            combined_thresh = cv2.bitwise_or(adaptive_thresh, otsu_thresh)
            combined_thresh = cv2.bitwise_or(combined_thresh, edges_dilated)
            
            # Find contours on the combined threshold
            contours, _ = cv2.findContours(combined_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Create mask for the shape
            mask = np.zeros_like(gray)
            if contours:
                # Filter contours by area to remove small noise
                min_area = (gray.shape[0] * gray.shape[1]) * 0.005
                valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
                
                if valid_contours:
                    # Find the largest contour
                    largest_contour = max(valid_contours, key=cv2.contourArea)
                    
                    # Get convex hull with refinement
                    hull = cv2.convexHull(largest_contour)
                    
                    # Approximate the contour with adaptive epsilon
                    perimeter = cv2.arcLength(hull, True)
                    epsilon = 0.01 * perimeter
                    approx_contour = cv2.approxPolyDP(hull, epsilon, True)
                    
                    # Create filled mask from the refined contour
                    cv2.drawContours(mask, [approx_contour], -1, 255, -1)
                    
                    # Apply morphological operations for final refinement
                    kernel = np.ones((5, 5), np.uint8)
                    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                    
                    # Additional processing for smoother edges
                    mask = cv2.GaussianBlur(mask, (5, 5), 0)
                    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            
            return mask
    
    def generate_3d_mesh_with_topology_optimization(self, image):
        """Generate 3D mesh with improved topology and smoother surfaces"""
        # Get shape mask with enhanced detection
        shape_mask = self.detect_shape(image)
        
        # Get enhanced depth map
        depth_map = self.estimate_depth(image, shape_mask, self.invert_depth)

        # Apply mask to depth map
        depth_map = depth_map * (shape_mask > 0).astype(float)
        
        # Enhanced smoothing of the depth map for better 3D appearance
        # Use guided filter which preserves edges better than bilateral filter
        if shape_mask.any():
            # Create a guidance image from the original for better edge preservation
            guide = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            depth_map = cv2.ximgproc.guidedFilter(guide, depth_map, 5, 0.01)
        
        # Create vertices only for the masked region
        height, width = depth_map.shape
        y, x = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        
        # Scale factors to convert to real dimensions if provided
        if self.real_dimensions:
            real_width, real_height, real_depth = self.real_dimensions
            scale_x = real_width / width
            scale_y = real_height / height
            scale_z = real_depth
        else:
            # Default scaling
            scale_x = 1.0 / max(width, height)
            scale_y = 1.0 / max(width, height)
            scale_z = self.extrusion_depth
            
        # Initialize vertices array
        vertices = []
        faces = []
        colors = []
        vertex_map = np.full((height, width), -1)  # Map to track vertex indices
        current_vertex = 0
        
        # Progress tracking
        total_pixels = height * width
        processed_pixels = 0
        last_percent = 0
        
        # Generate front face vertices only for masked region
        front_vertices_indices = {}
        for i in range(height):
            for j in range(width):
                processed_pixels += 1
                
                # Emit progress updates
                percent_complete = int((processed_pixels / total_pixels) * 50)  # Front face is 50% of work
                if percent_complete > last_percent:
                    self.progress.emit(percent_complete)
                    last_percent = percent_complete
                
                if shape_mask[i, j] > 0:
                    # Get depth value scaled by strength factor
                    z_value = depth_map[i, j] * scale_z
                    
                    # Create vertex with real dimensions
                    vertices.append([
                        (j - width/2) * scale_x,
                        (i - height/2) * scale_y,
                        z_value
                    ])
                    
                    vertex_map[i, j] = current_vertex
                    front_vertices_indices[(i, j)] = current_vertex
                    current_vertex += 1
        
        # Generate faces for front surface with improved topology
        # Use smaller quad cells to better capture details
        for i in range(height-1):
            for j in range(width-1):
                if (shape_mask[i,j] > 0 and shape_mask[i+1,j] > 0 and 
                    shape_mask[i,j+1] > 0 and shape_mask[i+1,j+1] > 0):
                    # Get vertex indices
                    v1 = vertex_map[i,j]
                    v2 = vertex_map[i,j+1]
                    v3 = vertex_map[i+1,j]
                    v4 = vertex_map[i+1,j+1]
                    
                    if all(v != -1 for v in [v1, v2, v3, v4]):
                        # Calculate which diagonal to use for better triangulation
                        # Compare depth values to create a more natural surface
                        z1 = depth_map[i, j]
                        z2 = depth_map[i, j+1]
                        z3 = depth_map[i+1, j]
                        z4 = depth_map[i+1, j+1]
                        
                        # Choose the diagonal that connects similar heights
                        # This creates a more natural surface topology
                        diag1_diff = abs(z1 - z4)
                        diag2_diff = abs(z2 - z3)
                        
                        if diag1_diff <= diag2_diff:
                            # Use diagonal v1-v4
                            faces.extend([[v1, v2, v4], [v1, v4, v3]])
                        else:
                            # Use diagonal v2-v3
                            faces.extend([[v1, v2, v3], [v2, v4, v3]])
                        
                        # Calculate color based on image with enhanced depth effect
                        base_color = image[i,j][:3] / 255.0  # Use only RGB channels
                        depth_factor = depth_map[i,j]
                        
                        # Enhanced color depth effect
                        ambient = 0.3  # Ambient light factor
                        diffuse = 0.7  # Diffuse light factor
                        color = base_color * (ambient + diffuse * depth_factor)
                        
                        # Add the two triangle colors
                        colors.extend([color, color])
        
        # If add_base is true, create a solid 3D object with improved side and back faces
        if self.add_base:
            # First, find the boundary pixels of the shape mask using contour detection
            # This creates a cleaner boundary than pixel-by-pixel checking
            contours, _ = cv2.findContours(shape_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            boundary_points = []
            if contours and len(contours[0]) > 2:
                # Get the largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Simplify the contour to reduce complexity while maintaining shape
                epsilon = 0.005 * cv2.arcLength(largest_contour, True)
                approx_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
                
                # Convert contour points to (i,j) coordinates
                for point in approx_contour:
                    j, i = point[0]  # x,y to column,row
                    if 0 <= i < height and 0 <= j < width:
                        boundary_points.append((i, j))
            
            # If no contour was found, fall back to pixel-based boundary detection
            if not boundary_points:
                # Use a more efficient method to find boundary points
                kernel = np.ones((3, 3), np.uint8)
                eroded = cv2.erode(shape_mask, kernel, iterations=1)
                boundary = shape_mask - eroded
                
                boundary_indices = np.where(boundary > 0)
                boundary_points = list(zip(boundary_indices[0], boundary_indices[1]))
                
                # Limit the number of boundary points for efficiency
                if len(boundary_points) > 500:
                    step = len(boundary_points) // 500
                    boundary_points = boundary_points[::step]
            
            # Sort boundary points to form a loop
            if boundary_points:
                # Create back face vertices (flat with slight curvature for better appearance)
                back_vertices_indices = {}
                base_z = -0.1 * scale_z  # Small offset for the base
                
                # Set a concave base for better stability
                base_center_z = -0.12 * scale_z
                
                total_boundary = len(boundary_points)
                for idx, (i, j) in enumerate(boundary_points):
                    # Calculate distance from center for base curvature
                    center_i, center_j = height // 2, width // 2
                    dist_from_center = np.sqrt(((i - center_i) / height) ** 2 + ((j - center_j) / width) ** 2)
                    # Create slight curvature in the base (concave)
                    back_z = base_z + (base_center_z - base_z) * (1 - dist_from_center) ** 2
                    
                    # Add back vertices
                    vertices.append([
                        (j - width/2) * scale_x,
                        (i - height/2) * scale_y,
                        back_z
                    ])
                    back_vertices_indices[(i, j)] = current_vertex
                    
                    # Add corresponding front vertex to create side faces
                    if (i, j) in front_vertices_indices:
                        front_idx = front_vertices_indices[(i, j)]
                        back_idx = current_vertex
                        
                        # Find next boundary point
                        next_idx = (idx + 1) % total_boundary
                        next_i, next_j = boundary_points[next_idx]
                        
                        if (next_i, next_j) in front_vertices_indices:
                            next_front_idx = front_vertices_indices[(next_i, next_j)]
                            next_back_idx = back_vertices_indices.get((next_i, next_j))
                            
                            if next_back_idx is None:
                                # Calculate back z value for next point
                                next_dist = np.sqrt(((next_i - center_i) / height) ** 2 + 
                                                ((next_j - center_j) / width) ** 2)
                                next_back_z = base_z + (base_center_z - base_z) * (1 - next_dist) ** 2
                                
                                # Add the back vertex for the next point
                                vertices.append([
                                    (next_j - width/2) * scale_x,
                                    (next_i - height/2) * scale_y,
                                    next_back_z
                                ])
                                next_back_idx = current_vertex + 1
                                back_vertices_indices[(next_i, next_j)] = next_back_idx
                                current_vertex += 1
                            
                            # Create side faces with improved orientation for better rendering
                            # Check the winding order to ensure correct face normals
                            faces.append([front_idx, next_front_idx, back_idx])
                            faces.append([next_front_idx, next_back_idx, back_idx])
                            
                            # Use a gradient color for sides based on the front face color
                            front_color = image[i,j][:3] / 255.0
                            side_color_top = front_color * 0.9  # Slightly darker than front
                            side_color_bottom = front_color * 0.6  # Darker for bottom
                            
                            colors.append(side_color_top)  # Top triangle
                            colors.append(side_color_bottom)  # Bottom triangle
                    
                    current_vertex += 1
                    
                    # Update progress
                    percent_complete = 50 + int((idx / total_boundary) * 40)  # Sides are 40% of work
                    if percent_complete > last_percent:
                        self.progress.emit(percent_complete)
                        last_percent = percent_complete
                
                # Create back face triangulation with improved method
                # Use a constrained Delaunay triangulation for a better base surface
                if len(back_vertices_indices) > 3:
                    # Get all back face vertices and indices
                    back_points = np.array([[j, i] for (i, j) in back_vertices_indices.keys()])
                    back_indices = list(back_vertices_indices.values())
                    
                    # Create a concave base centroid
                    centroid_i = np.mean(back_points[:, 1])
                    centroid_j = np.mean(back_points[:, 0])
                    center_dist = 0  # At centroid
                    centroid_z = base_z + (base_center_z - base_z) * (1 - center_dist) ** 2
                    
                    vertices.append([
                        (centroid_j - width/2) * scale_x,
                        (centroid_i - height/2) * scale_y,
                        centroid_z
                    ])
                    centroid_idx = current_vertex
                    current_vertex += 1
                    
                    # Create fan triangulation from centroid
                    for i in range(len(back_indices)):
                        v1 = back_indices[i]
                        v2 = back_indices[(i + 1) % len(back_indices)]
                        
                        # Check that we're not creating degenerate triangles
                        if v1 != v2 and v1 != centroid_idx and v2 != centroid_idx:
                            faces.append([centroid_idx, v1, v2])
                            
                            # Dark color for back face with slight variation for better visualization
                            darkness = 0.3 + 0.05 * np.random.random()  # Slight random variation
                            back_color = np.array([darkness, darkness, darkness * 1.1])  # Slightly blue tint
                            colors.append(back_color)
            
            # Final progress update
            self.progress.emit(100)
        
        # Convert to numpy arrays with proper types
        vertices = np.array(vertices, dtype=np.float32)
        faces = np.array(faces, dtype=np.uint32)
        colors = np.array(colors, dtype=np.float32)
        
        if len(faces) > 0:
            # Create mesh with proper data types
            mesh = trimesh.Trimesh(
                vertices=vertices,
                faces=faces,
                face_colors=(colors * 255).astype(np.uint8)
            )
            
            # Mesh cleanup and optimization with enhanced parameters
            mesh.remove_degenerate_faces()
            mesh.remove_duplicate_faces()
            mesh.remove_unreferenced_vertices()
            
            # Fix face winding order for consistent normals
            mesh.fix_normals()
            
            # Optional mesh smoothing for better appearance
            # Use Laplacian smoothing with a small lambda value to preserve details
            vertices_smoothed = mesh.vertices.copy()
            
            # Simple Laplacian smoothing implementation
            if len(mesh.vertices) > 0 and len(mesh.faces) > 0:
                adjacency = trimesh.graph.face_adjacency(mesh.faces)
                for _ in range(2):  # Two iterations of smoothing
                    for i in range(len(vertices_smoothed)):
                        # Find connected vertices
                        connected = []
                        for face in mesh.faces:
                            if i in face:
                                connected.extend([v for v in face if v != i])
                        
                        if connected:
                            # Remove duplicates
                            connected = list(set(connected))
                            # Calculate centroid of connected vertices
                            centroid = np.mean([vertices_smoothed[j] for j in connected], axis=0)
                            # Move vertex slightly toward centroid (lambda = 0.1)
                            vertices_smoothed[i] = vertices_smoothed[i] * 0.9 + centroid * 0.1
            
                # Update mesh with smoothed vertices
                # Only apply if the smoothing didn't create issues
                if not np.any(np.isnan(vertices_smoothed)):
                    mesh.vertices = vertices_smoothed
            
            # Center the mesh at origin
            mesh.vertices -= mesh.center_mass
            
            return mesh, mesh.vertices, mesh.faces, np.array(colors)
        else:
            return None, vertices, np.array([], dtype=np.uint32), np.array([], dtype=np.float32)
    def run(self):
        try:
            # Use the improved 3D mesh generation method
            mesh, vertices, faces, colors = self.generate_3d_mesh_with_topology_optimization(self.image)
            
            if mesh is not None:
                self.mesh_ready.emit(mesh)
                self.finished.emit(vertices, faces, colors)
            else:
                # Create an empty mesh if generation failed
                empty_mesh = trimesh.Trimesh()
                self.mesh_ready.emit(empty_mesh)
                self.finished.emit(np.array([]), np.array([]), np.array([]))
        except Exception as e:
            print(f"Error generating mesh: {str(e)}")
            # Create an empty mesh for error cases
            empty_mesh = trimesh.Trimesh()
            self.mesh_ready.emit(empty_mesh)
            self.finished.emit(np.array([]), np.array([]), np.array([]))
    def generate_enhanced_mesh(self, image):
        """Generate 3D mesh with real dimensions and solid base option"""
        # Detect shape and create mask
        shape_mask = self.detect_shape(image)
        
        # Get enhanced depth map using the shape mask
        depth_map = self.estimate_depth(image, shape_mask, self.invert_depth)
        
        # Apply mask to depth map
        depth_map = depth_map * (shape_mask > 0).astype(float)
        
        # Create vertices only for the masked region
        height, width = depth_map.shape
        y, x = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        
        # Scale factors to convert to real dimensions if provided
        if self.real_dimensions:
            real_width, real_height, real_depth = self.real_dimensions
            scale_x = real_width / width
            scale_y = real_height / height
            scale_z = real_depth
        else:
            # Default scaling
            scale_x = 1.0 / max(width, height)
            scale_y = 1.0 / max(width, height)
            scale_z = self.extrusion_depth
            
        # Initialize vertices array
        vertices = []
        faces = []
        colors = []
        vertex_map = np.full((height, width), -1)  # Map to track vertex indices
        current_vertex = 0
        
        # Progress tracking
        total_pixels = height * width
        processed_pixels = 0
        last_percent = 0
        
        # Generate front face vertices only for masked region
        front_vertices_indices = {}
        for i in range(height):
            for j in range(width):
                processed_pixels += 1
                
                # Emit progress updates
                percent_complete = int((processed_pixels / total_pixels) * 50)  # Front face is 50% of work
                if percent_complete > last_percent:
                    self.progress.emit(percent_complete)
                    last_percent = percent_complete
                
                if shape_mask[i, j] > 0:
                    # Get depth value scaled by strength factor
                    z_value = depth_map[i, j] * scale_z
                    
                    # Create vertex with real dimensions
                    vertices.append([
                        (j - width/2) * scale_x,
                        (i - height/2) * scale_y,
                        z_value
                    ])
                    
                    vertex_map[i, j] = current_vertex
                    front_vertices_indices[(i, j)] = current_vertex
                    current_vertex += 1
        
        # Generate faces for front surface
        for i in range(height-1):
            for j in range(width-1):
                if (shape_mask[i,j] > 0 and shape_mask[i+1,j] > 0 and 
                    shape_mask[i,j+1] > 0 and shape_mask[i+1,j+1] > 0):
                    # Get vertex indices
                    v1 = vertex_map[i,j]
                    v2 = vertex_map[i,j+1]
                    v3 = vertex_map[i+1,j]
                    v4 = vertex_map[i+1,j+1]
                    
                    if all(v != -1 for v in [v1, v2, v3, v4]):
                        # Create two triangles
                        faces.extend([[v1, v2, v3], [v3, v2, v4]])
                        
                        # Calculate color based on image and depth
                        base_color = image[i,j][:3] / 255.0  # Use only RGB channels
                        depth_factor = depth_map[i,j]
                        color = base_color * (0.7 + 0.3 * depth_factor)
                        colors.extend([color, color])
        
        # If add_base is true, create a solid 3D object by adding back face and sides
        if self.add_base:
            # First, find the boundary pixels of the shape mask
            boundary_points = []
            for i in range(height):
                for j in range(width):
                    if shape_mask[i, j] > 0:
                        # Check if this is a boundary pixel
                        is_boundary = False
                        for di, dj in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
                            ni, nj = i + di, j + dj
                            if (0 <= ni < height and 0 <= nj < width and shape_mask[ni, nj] == 0) or \
                               ni < 0 or ni >= height or nj < 0 or nj >= width:
                                is_boundary = True
                                break
                        
                        if is_boundary:
                            boundary_points.append((i, j))
            
            # Sort boundary points to form a loop (approximate)
            if boundary_points:
                # Create back face vertices (flat)
                back_vertices_indices = {}
                base_z = -0.1 * scale_z  # Small offset for the base
                
                total_boundary = len(boundary_points)
                for idx, (i, j) in enumerate(boundary_points):
                    # Add back vertices
                    vertices.append([
                        (j - width/2) * scale_x,
                        (i - height/2) * scale_y,
                        base_z
                    ])
                    back_vertices_indices[(i, j)] = current_vertex
                    
                    # Add corresponding front vertex to create side faces
                    if (i, j) in front_vertices_indices:
                        front_idx = front_vertices_indices[(i, j)]
                        back_idx = current_vertex
                        
                        # Find next boundary point (approximate)
                        next_idx = (idx + 1) % total_boundary
                        next_i, next_j = boundary_points[next_idx]
                        
                        if (next_i, next_j) in front_vertices_indices:
                            next_front_idx = front_vertices_indices[(next_i, next_j)]
                            next_back_idx = back_vertices_indices.get((next_i, next_j))
                            
                            if next_back_idx is None:
                                # Add the back vertex for the next point if not created yet
                                vertices.append([
                                    (next_j - width/2) * scale_x,
                                    (next_i - height/2) * scale_y,
                                    base_z
                                ])
                                next_back_idx = current_vertex + 1
                                back_vertices_indices[(next_i, next_j)] = next_back_idx
                                current_vertex += 1
                            
                            # Create side faces (two triangles)
                            faces.append([front_idx, next_front_idx, back_idx])
                            faces.append([next_front_idx, next_back_idx, back_idx])
                            
                            # Use a darker version of the front face color
                            side_color = np.array(image[i,j][:3] / 255.0) * 0.7
                            colors.extend([side_color, side_color])
                    
                    current_vertex += 1
                    
                    # Update progress
                    percent_complete = 50 + int((idx / total_boundary) * 40)  # Sides are 40% of work
                    if percent_complete > last_percent:
                        self.progress.emit(percent_complete)
                        last_percent = percent_complete
                
                # Create back face triangulation
                # Use a simplified approach by triangulating from the centroid
                back_vertices = [vertices[idx] for idx in back_vertices_indices.values()]
                if back_vertices:
                    centroid = np.mean(back_vertices, axis=0)
                    vertices.append(centroid)
                    centroid_idx = current_vertex
                    current_vertex += 1
                    
                    # Create triangles from centroid to each edge
                    back_indices = list(back_vertices_indices.values())
                    for i in range(len(back_indices)):
                        v1 = back_indices[i]
                        v2 = back_indices[(i + 1) % len(back_indices)]
                        faces.append([centroid_idx, v1, v2])
                        
                        # Dark color for back face
                        back_color = np.array([0.3, 0.3, 0.3])
                        colors.append(back_color)
            
            # Final progress update
            self.progress.emit(100)
        
        # Convert to numpy arrays with proper types
        vertices = np.array(vertices, dtype=np.float32)
        faces = np.array(faces, dtype=np.uint32)
        colors = np.array(colors, dtype=np.float32)
        
        if len(faces) > 0:
            # Create mesh with proper data types
            mesh = trimesh.Trimesh(
                vertices=vertices,
                faces=faces,
                face_colors=(colors * 255).astype(np.uint8)
            )
            
            # Mesh cleanup and optimization
            mesh.remove_degenerate_faces()
            mesh.remove_duplicate_faces()
            mesh.remove_unreferenced_vertices()
            
            # Center the mesh at origin
            mesh.vertices -= mesh.center_mass
            
            # Check if we need to apply real dimensions scaling
            if self.real_dimensions:
                # The mesh is already scaled during creation
                pass
            
            return mesh, mesh.vertices, mesh.faces, np.array(colors)
        else:
            return None, vertices, np.array([], dtype=np.uint32), np.array([], dtype=np.float32)

    def run(self):
        try:
            mesh, vertices, faces, colors = self.generate_enhanced_mesh(self.image)
            if mesh is not None:
                self.mesh_ready.emit(mesh)
                self.finished.emit(vertices, faces, colors)
            else:
                # Create an empty mesh if generation failed
                empty_mesh = trimesh.Trimesh()
                self.mesh_ready.emit(empty_mesh)
                self.finished.emit(np.array([]), np.array([]), np.array([]))
        except Exception as e:
            print(f"Error generating mesh: {str(e)}")
            # Create an empty mesh for error cases
            empty_mesh = trimesh.Trimesh()
            self.mesh_ready.emit(empty_mesh)
            self.finished.emit(np.array([]), np.array([]), np.array([]))

class RemoveBackgroundThread(QThread):
    """Thread for removing background"""
    finished = pyqtSignal(np.ndarray, np.ndarray)
    error = pyqtSignal(str)
    
    def __init__(self, bg_remover, image):
        super().__init__()
        self.bg_remover = bg_remover
        self.image = image
    
    def run(self):
        try:
            result_image, mask = self.bg_remover.remove_background(self.image)
            self.finished.emit(result_image, mask)
        except Exception as e:
            self.error.emit(str(e))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OneUp:Converting 2D Images Into 3D Model Through Digital Image Processing")
        self.setGeometry(100, 100, 1300, 800)
        self.setMinimumSize(1200, 700)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            QWidget {
                font-family: Arial, sans-serif;
                font-size: 12px;
                color: #ffffff;
            }
            QLabel {
                color: #ffffff;
            }
            QGroupBox {
                color: #ffffff;
            }
            QSlider {
                height: 20px;
            }
            QSlider::groove:horizontal {
                border: 1px solid #999999;
                height: 8px;
                background: #2d2d2d;
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #5c5c5c;
                border: 1px solid #5c5c5c;
                width: 18px;
                height: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
            QSlider::handle:horizontal:hover {
                background: #808080;
            }
            QProgressBar {
                border: 1px solid #555555;
                border-radius: 4px;
                background-color: #2d2d2d;
                text-align: center;
                color: white;
            }
            QProgressBar::chunk {
                background-color: #3498db;
                border-radius: 3px;
            }
            QCheckBox {
                spacing: 10px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
            QDoubleSpinBox {
                background-color: #2d2d2d;
                color: white;
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 4px;
            }
        """)
        
        # Initialize background removal
        self.init_bg_removal()
        
        self.init_ui()
        self.image_path = None
        self.current_mesh = None

    def init_bg_removal(self):
        """Initialize background removal feature"""
        self.bg_remover = RembgBackgroundRemover()
        self.original_image = None
        self.processed_image = None
        self.mask = None
        
        # Create button style as property for reuse
        self.button_style = """
            QPushButton {
                background-color: #363636;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #404040;
            }
            QPushButton:pressed {
                background-color: #505050;
            }
            QPushButton:disabled {
                background-color: #2a2a2a;
                color: #606060;
            }
        """

    def add_background_removal_ui(self):
        # Background removal group
        bg_removal_group = QGroupBox("Background Removal")
        bg_removal_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #404040;
                border-radius: 6px;
                margin-top: 6px;
                padding-top: 10px;
                background-color: #2d2d2d;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #ffffff;
            }
        """)
        bg_removal_layout = QVBoxLayout(bg_removal_group)
        
        # Remove background button
        self.remove_bg_button = QPushButton("🎭 Remove Background")
        self.remove_bg_button.setMinimumHeight(40)
        self.remove_bg_button.setEnabled(False)
        self.remove_bg_button.setStyleSheet(self.button_style)
        self.remove_bg_button.clicked.connect(self.remove_background)
        
        # Reset button
        self.reset_image_button = QPushButton("↩️ Reset Image")
        self.reset_image_button.setMinimumHeight(40)
        self.reset_image_button.setEnabled(False)
        self.reset_image_button.setStyleSheet(self.button_style)
        self.reset_image_button.clicked.connect(self.reset_image)
        
        # Add to layout
        bg_removal_layout.addWidget(self.remove_bg_button)
        bg_removal_layout.addWidget(self.reset_image_button)
        
        return bg_removal_group

    def add_3d_settings_ui(self):
        # 3D Settings group
        settings_group = QGroupBox("3D Generation Settings")
        settings_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #404040;
                border-radius: 6px;
                margin-top: 6px;
                padding-top: 10px;
                background-color: #2d2d2d;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #ffffff;
            }
        """)
        
        settings_layout = QFormLayout(settings_group)
        settings_layout.setContentsMargins(10, 25, 10, 10)
        settings_layout.setSpacing(15)
        
        # Depth strength slider
        self.depth_slider = QSlider(Qt.Orientation.Horizontal)
        self.depth_slider.setMinimum(10)
        self.depth_slider.setMaximum(300)
        self.depth_slider.setValue(100)
        self.depth_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.depth_slider.setTickInterval(50)
        
        # Extrusion depth slider
        self.extrusion_slider = QSlider(Qt.Orientation.Horizontal)
        self.extrusion_slider.setMinimum(10)
        self.extrusion_slider.setMaximum(200)
        self.extrusion_slider.setValue(50)
        self.extrusion_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.extrusion_slider.setTickInterval(25)
        
        # Add solid base checkbox
        self.add_base_checkbox = QCheckBox("Create Solid 3D Object")
        self.add_base_checkbox.setChecked(True)
        
        # NEW: Add invert depth checkbox
        self.invert_depth_checkbox = QCheckBox("Invert Depth")
        self.invert_depth_checkbox.setChecked(True)  # Default to inverted depth
        self.invert_depth_checkbox.setToolTip("When checked, dark areas will be raised. When unchecked, light areas will be raised.")
        
        # Create a helper text label
        self.invert_depth_label = QLabel("• Checked: Text and details pop up\n• Unchecked: Outlines and borders pop up")
        self.invert_depth_label.setStyleSheet("color: #999999; font-size: 10px; margin-left: 25px;")
        
        # Add real dimensions options
        dimensions_layout = QHBoxLayout()
        
        # Width input
        self.width_input = QDoubleSpinBox()
        self.width_input.setRange(1, 1000)
        self.width_input.setValue(100)
        self.width_input.setSuffix(" mm")
        
        # Height input
        self.height_input = QDoubleSpinBox()
        self.height_input.setRange(1, 1000)
        self.height_input.setValue(100)
        self.height_input.setSuffix(" mm")
        
        # Depth input
        self.depth_input = QDoubleSpinBox()
        self.depth_input.setRange(1, 1000)
        self.depth_input.setValue(50)
        self.depth_input.setSuffix(" mm")
        
        # Add to dimensions layout
        dimensions_layout.addWidget(QLabel("W:"))
        dimensions_layout.addWidget(self.width_input)
        dimensions_layout.addWidget(QLabel("H:"))
        dimensions_layout.addWidget(self.height_input)
        dimensions_layout.addWidget(QLabel("D:"))
        dimensions_layout.addWidget(self.depth_input)
        
        # Use real dimensions checkbox
        self.use_real_dims_checkbox = QCheckBox("Use Real Dimensions")
        self.use_real_dims_checkbox.setChecked(True)
        
        # Add to settings layout
        settings_layout.addRow("Depth Strength:", self.depth_slider)
        settings_layout.addRow("Extrusion Depth:", self.extrusion_slider)
        settings_layout.addRow("", self.add_base_checkbox)
        settings_layout.addRow("", self.invert_depth_checkbox)
        settings_layout.addRow("", self.invert_depth_label)
        settings_layout.addRow("", self.use_real_dims_checkbox)
        settings_layout.addRow("Dimensions:", dimensions_layout)
        
        return settings_group

    def init_ui(self):
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        # Create horizontal layout for main content
        content_layout = QHBoxLayout()
        
        # Left panel for image and controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(15)
        
        # Image preview group
        image_group = QGroupBox("Image Preview")
        image_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #404040;
                border-radius: 6px;
                margin-top: 6px;
                padding-top: 10px;
                background-color: #2d2d2d;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #ffffff;
            }
        """)
        image_layout = QVBoxLayout(image_group)
        
        self.image_label = QLabel("No image selected")
        self.image_label.setFixedSize(400, 300)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("""
            QLabel {
                background-color: #1e1e1e;
                border: 2px dashed #404040;
                border-radius: 4px;
                color: #808080;
            }
        """)
        image_layout.addWidget(self.image_label)
        
        # Background removal group
        bg_removal_group = self.add_background_removal_ui()
        
        # 3D settings group
        settings_group = self.add_3d_settings_ui()
        
        # Controls group
        controls_group = QGroupBox("Controls")
        controls_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #404040;
                border-radius: 6px;
                margin-top: 6px;
                padding-top: 10px;
                background-color: #2d2d2d;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #ffffff;
            }
        """)
        controls_layout = QVBoxLayout(controls_group)
        
        self.select_button = QPushButton("📁 Select Image")
        self.select_button.setMinimumHeight(40)
        self.select_button.setStyleSheet(self.button_style)
        self.select_button.clicked.connect(self.select_image_with_bg_removal)
        
        self.convert_button = QPushButton("🔄 Generate 3D Model")
        self.convert_button.setMinimumHeight(40)
        self.convert_button.setEnabled(False)
        self.convert_button.setStyleSheet(self.button_style)
        self.convert_button.clicked.connect(self.convert_to_3d)
        
        self.export_button = QPushButton("💾 Export 3D Model")
        self.export_button.setMinimumHeight(40)
        self.export_button.setEnabled(False)
        self.export_button.setStyleSheet(self.button_style)
        self.export_button.clicked.connect(self.export_mesh)
        
        # Add progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p% Complete")
        
        controls_layout.addWidget(self.select_button)
        controls_layout.addWidget(self.convert_button)
        controls_layout.addWidget(self.progress_bar)
        controls_layout.addWidget(self.export_button)
        
        # Add groups to left panel
        left_layout.addWidget(image_group)
        left_layout.addWidget(bg_removal_group)
        left_layout.addWidget(settings_group)
        left_layout.addWidget(controls_group)
        
        # Right panel for 3D viewer
        viewer_group = QGroupBox("3D Preview")
        viewer_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #404040;
                border-radius: 6px;
                margin-top: 6px;
                padding-top: 10px;
                background-color: #2d2d2d;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #ffffff;
            }
        """)
        viewer_layout = QVBoxLayout(viewer_group)
        
        self.viewer = gl.GLViewWidget()
        self.viewer.setMinimumSize(600, 600)
        self.viewer.setCameraPosition(distance=40, elevation=30, azimuth=45)
        self.viewer.setBackgroundColor('#1e1e1e')  # Dark background
        
        # Add grid for better perspective
        grid = gl.GLGridItem()
        grid.setSize(x=100, y=100, z=1)
        grid.setSpacing(x=10, y=10, z=10)
        grid.setColor((0.3, 0.3, 0.3, 1.0))  # Dark gray grid
        self.viewer.addItem(grid)
        
        # Add controls for 3D view
        view_controls = QHBoxLayout()
        
        # Add rotation buttons
        self.rotate_x_button = QPushButton("Rotate X")
        self.rotate_x_button.setStyleSheet(self.button_style)
        self.rotate_x_button.clicked.connect(lambda: self.rotate_view(90, 0, 0))
        
        self.rotate_y_button = QPushButton("Rotate Y")
        self.rotate_y_button.setStyleSheet(self.button_style)
        self.rotate_y_button.clicked.connect(lambda: self.rotate_view(0, 90, 0))
        
        self.rotate_z_button = QPushButton("Rotate Z")
        self.rotate_z_button.setStyleSheet(self.button_style)
        self.rotate_z_button.clicked.connect(lambda: self.rotate_view(0, 0, 90))
        
        self.reset_view_button = QPushButton("Reset View")
        self.reset_view_button.setStyleSheet(self.button_style)
        self.reset_view_button.clicked.connect(self.reset_view)
        
        view_controls.addWidget(self.rotate_x_button)
        view_controls.addWidget(self.rotate_y_button)
        view_controls.addWidget(self.rotate_z_button)
        view_controls.addWidget(self.reset_view_button)
        
        viewer_layout.addWidget(self.viewer)
        viewer_layout.addLayout(view_controls)
        
        # Add panels to content layout
        content_layout.addWidget(left_panel, 1)
        content_layout.addWidget(viewer_group, 2)
        
        # Add content layout to main layout
        main_layout.addLayout(content_layout)
        
        self.setCentralWidget(main_widget)
        
    def rotate_view(self, x, y, z):
        """Rotate the 3D view"""
        if hasattr(self, 'mesh_item'):
            self.mesh_item.rotate(x, y, z)
            self.viewer.update()
    
    def reset_view(self):
        """Reset the 3D view to default"""
        self.viewer.setCameraPosition(distance=40, elevation=30, azimuth=45)
        self.viewer.update()

    def select_image_with_bg_removal(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )
        if file_name:
            self.image_path = file_name
            
            # Reset processed image
            self.processed_image = None
            self.original_image = None
            
            # Load and display image
            pixmap = QPixmap(file_name)
            self.image_label.setPixmap(pixmap.scaled(
                self.image_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            ))
            
            # Enable buttons
            self.convert_button.setEnabled(True)
            self.remove_bg_button.setEnabled(True)
            self.reset_image_button.setEnabled(False)
            
            # Reset progress bar
            self.progress_bar.setValue(0)

    def background_removal_finished(self, result_image, mask):
        """Callback for when background removal finishes"""
        self.processed_image = result_image
        self.mask = mask
        
        # Convert BGRA to QPixmap
        height, width, channel = self.processed_image.shape
        bytes_per_line = 4 * width
        q_img = QImage(self.processed_image.data, width, height, bytes_per_line, QImage.Format.Format_RGBA8888)
        pixmap = QPixmap.fromImage(q_img)
        
        # Display the processed image
        self.image_label.setPixmap(pixmap.scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        ))
        
        # Enable reset button
        self.reset_image_button.setEnabled(True)
        self.remove_bg_button.setEnabled(True)  # Re-enable the button
        self.remove_bg_button.setText("🎭 Remove Background")  # Reset text

    def background_removal_error(self, error_message):
        """Callback for background removal errors"""
        QMessageBox.warning(
            self,
            "Background Removal Error",
            f"Failed to remove background: {error_message}\n\n"
            "Please try again or use a different image."
        )
        self.remove_bg_button.setEnabled(True)  # Re-enable the button
        self.remove_bg_button.setText("🎭 Remove Background")  # Reset text

    def remove_background(self):
        """Remove background using"""
        if not self.image_path:
            return
        
        # Get the current image
        if hasattr(self, 'original_image') and self.original_image is not None:
            image = self.original_image.copy()
        else:
            image = cv2.imread(self.image_path)
            self.original_image = image.copy()
        
        if image is None:
            QMessageBox.warning(
                self,
                "Error",
                "Failed to load image. Please select a different image."
            )
            return
        
        # Disable button during processing
        self.remove_bg_button.setEnabled(False)
        self.remove_bg_button.setText("Removing Background...")
        
        # Create and start background thread
        self.bg_thread = RemoveBackgroundThread(self.bg_remover, image)
        self.bg_thread.finished.connect(self.background_removal_finished)
        self.bg_thread.error.connect(self.background_removal_error)
        self.bg_thread.start()

    def reset_image(self):
        """Reset to the original image"""
        if hasattr(self, 'original_image') and self.original_image is not None:
            # Convert BGR to QPixmap
            height, width, channel = self.original_image.shape
            bytes_per_line = 3 * width
            q_img = QImage(self.original_image.data, width, height, bytes_per_line, QImage.Format.Format_BGR888)
            pixmap = QPixmap.fromImage(q_img)
            
            # Display the original image
            self.image_label.setPixmap(pixmap.scaled(
                self.image_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            ))
            
            # Reset processed image
            self.processed_image = None
            
            # Disable reset button
            self.reset_image_button.setEnabled(False)

    def update_progress(self, value):
        """Update progress bar value"""
        self.progress_bar.setValue(value)
        QApplication.processEvents()  # Ensure UI updates

    def convert_to_3d(self):
        """Convert image to 3D with enhanced options"""
        if not self.image_path:
            return
            
        try:
            # Use processed image if available, otherwise use original
            if hasattr(self, 'processed_image') and self.processed_image is not None:
                # If BGRA, keep alpha channel for better shape detection
                if self.processed_image.shape[2] == 4:
                    image = self.processed_image.copy()
                else:
                    image = self.processed_image.copy()
            else:
                image = cv2.imread(self.image_path)
            
            if image is None:
                raise ValueError("Failed to load image")
                
            # Disable convert button during processing
            self.convert_button.setEnabled(False)
            self.convert_button.setText("Generating 3D Model...")
            self.progress_bar.setValue(0)
            
            # Get settings
            depth_strength = self.depth_slider.value() / 100.0
            extrusion_depth = self.extrusion_slider.value() / 100.0
            add_base = self.add_base_checkbox.isChecked()
            invert_depth = self.invert_depth_checkbox.isChecked()  # Get the invert depth setting
            
            # Check if using real dimensions
            if self.use_real_dims_checkbox.isChecked():
                real_dimensions = (
                    self.width_input.value(),
                    self.height_input.value(),
                    self.depth_input.value() / 100.0  # Scale down depth
                )
            else:
                real_dimensions = None
            
            # Downscale for better performance
            max_size = 256  # Larger size for better detail
            height, width = image.shape[:2]
            scale = max_size / max(height, width)
            new_size = (int(width * scale), int(height * scale))
            image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
            
            # Start mesh generation with settings including invert_depth option
            self.mesh_thread = EnhancedMeshGenerator(
                image, 
                depth_strength=depth_strength,
                extrusion_depth=extrusion_depth,
                add_base=add_base,
                invert_depth=invert_depth,  # Pass the invert depth setting
                real_dimensions=real_dimensions
            )
            
            self.mesh_thread.progress.connect(self.update_progress)
            self.mesh_thread.finished.connect(self.display_mesh)
            self.mesh_thread.mesh_ready.connect(self.store_mesh)
            self.mesh_thread.start()
            
        except Exception as e:
            print(f"Error in convert_to_3d: {str(e)}")
            QMessageBox.critical(
                self,
                "Conversion Error",
                "Failed to process the image. Please try a different image."
            )
            self.convert_button.setEnabled(True)
            self.convert_button.setText("🔄 Generate 3D Model")

    def display_mesh(self, vertices, faces, colors):
        """Display the 3D mesh with enhanced rendering and proper centering"""
        # Import QtGui for QVector3D
        from PyQt6 import QtGui
        
        self.viewer.clear()
        
        try:
            if len(faces) > 0 and len(vertices) > 0:
                # Basic data validation
                if len(vertices) < 3 or len(faces) < 1:
                    raise ValueError("Not enough vertices or faces")

                # Ensure proper data types and shapes
                vertices = np.array(vertices, dtype=np.float32)
                faces = np.array(faces, dtype=np.uint32)
                
                # Calculate the center of the mesh
                center = np.mean(vertices, axis=0)
                
                # Center the vertices by subtracting the center point
                centered_vertices = vertices - center
                
                # Create colors if missing
                if len(colors) != len(faces) or colors.shape[1] != 3:
                    colors = np.ones((len(faces), 4), dtype=np.float32) * [0.7, 0.7, 0.7, 1.0]
                else:
                    # Convert RGB to RGBA
                    alpha = np.ones((len(colors), 1), dtype=np.float32)
                    colors = np.hstack([colors, alpha]).astype(np.float32)

                # Create mesh with enhanced parameters and centered vertices
                self.mesh_item = gl.GLMeshItem(
                    vertexes=centered_vertices,
                    faces=faces,
                    faceColors=colors,
                    smooth=True,  # Enable smooth shading
                    computeNormals=True,  # Compute normals for better lighting
                    drawEdges=False,  # Hide edges for smoother appearance
                    shader='shaded'  # Use shaded rendering for better 3D effect
                )
                
                # Add reference grid
                grid = gl.GLGridItem()
                grid.setSize(x=2, y=2, z=0.1)
                grid.setSpacing(x=0.1, y=0.1, z=0.1)
                
                # Adjust grid position to be below the mesh
                min_z = centered_vertices[:, 2].min()
                grid.translate(0, 0, min_z - 0.1)
                
                # Add axis for reference
                x_axis = gl.GLLinePlotItem(pos=np.array([[0,0,0], [1,0,0]]), color=(1,0,0,1), width=2)
                y_axis = gl.GLLinePlotItem(pos=np.array([[0,0,0], [0,1,0]]), color=(0,1,0,1), width=2)
                z_axis = gl.GLLinePlotItem(pos=np.array([[0,0,0], [0,0,1]]), color=(0,0,1,1), width=2)
                
                # Add items to viewer
                self.viewer.addItem(grid)
                self.viewer.addItem(x_axis)
                self.viewer.addItem(y_axis)
                self.viewer.addItem(z_axis)
                self.viewer.addItem(self.mesh_item)
                
                # Determine appropriate camera distance based on mesh size
                mesh_size = np.max(np.ptp(centered_vertices, axis=0))
                camera_distance = max(3.0, mesh_size * 1.5)  # Ensure minimum distance of 3.0
                
                # Set camera position with appropriate distance
                self.viewer.setCameraPosition(distance=camera_distance, elevation=30, azimuth=45)
                
                # Set rendering options
                self.viewer.opts['distance'] = camera_distance
                self.viewer.opts['fov'] = 60
                self.viewer.opts['elevation'] = 30
                self.viewer.opts['azimuth'] = 45
                
                # Reset camera target to origin (0,0,0) where mesh is now centered
                self.viewer.opts['center'] = QtGui.QVector3D(0, 0, 0)
                
                # Update view
                self.viewer.update()
                
                # Show success message
                QMessageBox.information(
                    self,
                    "Success",
                    "3D model generated successfully!"
                )
                
        except Exception as e:
            print(f"Error displaying mesh: {str(e)}")
            QMessageBox.warning(
                self,
                "Display Error",
                "Failed to display the 3D model. Please try with a different image."
            )
        
        finally:
            # Re-enable convert button
            self.convert_button.setEnabled(True)
            self.convert_button.setText("🔄 Generate 3D Model")
    
    def export_mesh(self):
        """Export the current mesh to a file with enhanced options"""
        if self.current_mesh is None:
            QMessageBox.warning(
                self,
                "Export Error",
                "No 3D model available to export. Please generate a model first."
            )
            return
                
        try:
            file_name, file_type = QFileDialog.getSaveFileName(
                self,
                "Save 3D Model",
                "",
                "STL Files (*.stl);;OBJ Files (*.obj);;PLY Files (*.ply)"
            )
            
            if file_name:
                # Add extension if not present
                if file_type == "STL Files (*.stl)" and not file_name.lower().endswith('.stl'):
                    file_name += '.stl'
                elif file_type == "OBJ Files (*.obj)" and not file_name.lower().endswith('.obj'):
                    file_name += '.obj'
                elif file_type == "PLY Files (*.ply)" and not file_name.lower().endswith('.ply'):
                    file_name += '.ply'
                
                # Export the mesh
                self.current_mesh.export(file_name)
                
                QMessageBox.information(
                    self,
                    "Success",
                    f"Model exported successfully to:\n{file_name}"
                )
                
        except Exception as e:
            print(f"Error exporting mesh: {str(e)}")
            QMessageBox.critical(
                self,
                "Export Error",
                f"Failed to export the model: {str(e)}\nPlease try again with a different format or location."
            )

    def store_mesh(self, mesh):
        """Store the generated mesh and enable export if valid"""
        self.current_mesh = mesh
        if mesh is not None and len(mesh.faces) > 0:
            self.export_button.setEnabled(True)
        else:
            self.export_button.setEnabled(False)
            print("Warning: Invalid or empty mesh received")


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()