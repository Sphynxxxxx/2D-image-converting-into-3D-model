import sys
import cv2
import numpy as np
import trimesh
from PIL import Image
from rembg import remove
import math
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                           QPushButton, QLabel, QFileDialog, QMessageBox, QSlider, QCheckBox,
                           QGroupBox, QSizePolicy)
from PyQt6.QtGui import QPixmap, QImage, QCursor, QFont
import pyqtgraph.opengl as gl

class Shape3DConverter:
    def __init__(self):
        self.circle_segments = 72  # Segments for circle approximation
        self.sphere_segments = 64  # Segments for sphere generation
        self.true_3d_mode = False  # Default to extrusion mode
        self.heart_3d_mode = False  # Special mode for 3D hearts
        self.vertex_radius = 0.0   # Default vertex radius (0 means no vertices shown)
        self.smooth_heart = True  # Heart smoothing option - still keeping the property with default value
        self.smoothing_factor = 0.0  # Initialize smoothing factor
        self.inflation_enabled = False  # New property for inflation mode
        self.inflation_factor = 0.5  # Default inflation factor (0.0 to 1.0)
        self.inflation_distribution = 0.0
        self.extrusion_strength = 1.0  # Default extrusion strength multiplier


    def set_extrusion_strength(self, strength):
        """Set the extrusion strength multiplier (0.1 to 3.0)"""
        self.extrusion_strength = max(0.1, min(3.0, strength))

    def set_inflation_enabled(self, enabled):
        """Enable or disable shape inflation mode"""
        self.inflation_enabled = enabled
        
    def set_inflation_factor(self, factor):
        """Set inflation factor (0.0 to 1.0)"""
        self.inflation_factor = max(0.0, min(1.0, factor))

    def set_smoothing_factor(self, factor):
        """Set edge smoothing factor (0.0 to 1.0)"""
        self.smoothing_factor = factor

    def set_true_3d_mode(self, enabled):
        """Enable or disable true 3D mode"""
        self.true_3d_mode = enabled
        if enabled:
            self.heart_3d_mode = False
        
    def set_heart_3d_mode(self, enabled):
        """Enable or disable specialized heart 3D mode"""
        self.heart_3d_mode = enabled
        if enabled:
            self.true_3d_mode = False
        
    def set_vertex_radius(self, radius):
        """Set the radius of vertices in the mesh"""
        self.vertex_radius = radius

    def remove_background(self, image):
        """Remove background using rembg"""
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        output = remove(pil_img)
        output_array = np.array(output)
        bgra = cv2.cvtColor(output_array, cv2.COLOR_RGBA2BGRA)
        mask = (bgra[:, :, 3] > 0).astype(np.uint8) * 255
        return bgra, mask

    def inflate_mesh(self, vertices, faces, colors):
        """
        Create a balloon-like inflated version of a mesh with uniform distribution
        and protection for center points
        """
        if not vertices or not faces or len(vertices) < 3:
            return vertices, faces, colors
        
        try:
            vertices = np.array(vertices, dtype=np.float32)
            faces = np.array(faces, dtype=np.int32)
            
            # Create a trimesh object
            temp_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            
            # Identify center vertices (vertices connected to many faces)
            vertex_face_count = np.zeros(len(vertices), dtype=np.int32)
            for face in faces:
                for vertex in face:
                    vertex_face_count[vertex] += 1
                    
            # Centers typically have many more connections
            center_indices = np.where(vertex_face_count > np.mean(vertex_face_count) * 1.5)[0]
            
            # Store original heights of center vertices
            center_heights = {}
            for idx in center_indices:
                center_heights[idx] = vertices[idx, 2]
            
            # Calculate mesh properties for scaling
            mesh_size = np.max(np.ptp(vertices, axis=0))
            inflation_distance = mesh_size * 0.8 * self.inflation_factor
            
            # Get vertex normals
            vertex_normals = temp_mesh.vertex_normals
            
            # Calculate centroid of the mesh
            centroid = np.mean(vertices, axis=0)
            
            # 1. Center-based inflation but with more uniform distribution
            vectors_from_center = vertices - centroid
            distances_from_center = np.linalg.norm(vectors_from_center, axis=1)
            max_distance = np.max(distances_from_center)
            
            # Normalize distances
            normalized_distances = distances_from_center / max_distance
            
            # Use different scaling strategies based on inflation_distribution setting
            if self.inflation_distribution > 0:
                # More inflation at center
                power = 2.0 - self.inflation_distribution
                scale_factors = 1.0 + inflation_distance * (1.0 - normalized_distances ** power)
            elif self.inflation_distribution < 0:
                # More inflation at edges
                power = 2.0 + abs(self.inflation_distribution)
                scale_factors = 1.0 + inflation_distance * normalized_distances ** power
            else:
                # Uniform inflation
                scale_factors = 1.0 + inflation_distance * 0.5 * (1.0 + normalized_distances)
            
            # Apply center-based scaling with more uniformity
            inflated_vertices = centroid + vectors_from_center * scale_factors[:, np.newaxis]
            
            # 2. Normal-based smoothing (rounds out the shape)
            # Apply normal-based inflation more uniformly
            normal_strength = inflation_distance * 0.3
            inflated_vertices += vertex_normals * normal_strength
            
            # Apply Laplacian smoothing for a more even surface
            smooth_mesh = trimesh.Trimesh(vertices=inflated_vertices, faces=faces)
            trimesh.smoothing.filter_laplacian(smooth_mesh, iterations=4)
            inflated_vertices = smooth_mesh.vertices
            
            # Restore original heights for center vertices
            for idx, height in center_heights.items():
                # Adjust center vertex height while maintaining some inflation effect
                # (we don't want to completely flatten it)
                current_height = inflated_vertices[idx, 2]
                # Keep 80% of the original height plus 20% of the inflated height
                inflated_vertices[idx, 2] = height * 0.8 + current_height * 0.2
            
            return inflated_vertices.tolist(), faces.tolist(), colors
            
        except Exception as e:
            print(f"Inflation error: {e}")
            import traceback
            traceback.print_exc()
            return vertices, faces, colors
        
    def taubin_smooth_mesh(self, mesh, iterations=5):
        """Apply Taubin smoothing to mesh for better quality than Laplacian smoothing
        
        Taubin smoothing prevents shrinkage by using alternating positive and negative smoothing factors
        """
        try:
            # Make a copy to avoid modifying original
            smoothed = mesh.copy()
            
            # Smoothing parameters
            lamb = 0.5    # Positive smoothing factor
            mu = -0.53    # Negative smoothing factor
            
            # Get vertex neighbors from mesh
            new_vertices = np.array(smoothed.vertices, dtype=np.float64)
            adjacency = smoothed.vertex_neighbors
            
            # Apply multiple iterations of smoothing
            for _ in range(iterations):
                # First pass (inflation)
                offsets = np.zeros_like(new_vertices)
                for i, neighbors in enumerate(adjacency):
                    if neighbors:
                        neighbor_verts = new_vertices[neighbors]
                        average = np.mean(neighbor_verts, axis=0)
                        offsets[i] = lamb * (average - new_vertices[i])
                new_vertices += offsets
                
                # Second pass (anti-inflation)
                offsets = np.zeros_like(new_vertices)
                for i, neighbors in enumerate(adjacency):
                    if neighbors:
                        neighbor_verts = new_vertices[neighbors]
                        average = np.mean(neighbor_verts, axis=0)
                        offsets[i] = mu * (average - new_vertices[i])
                new_vertices += offsets
                
            # Update mesh vertices
            smoothed.vertices = new_vertices
            return smoothed
            
        except Exception as e:
            print(f"Taubin smoothing failed: {e}")
            import traceback
            traceback.print_exc()
            return mesh

    def get_contour_color(self, contour, image):
        """Get the dominant color within a contour with improved accuracy"""
        try:
            # Create a mask for the contour
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            
            # Check if we have an alpha channel (4 channels)
            if image.shape[2] == 4:
                # For BGRA images, only consider pixels that aren't transparent
                alpha_mask = image[:,:,3] > 128
                combined_mask = cv2.bitwise_and(mask, mask, mask=alpha_mask.astype(np.uint8))
            else:
                combined_mask = mask
            
            # If no valid pixels after masking, return default
            if cv2.countNonZero(combined_mask) == 0:
                return [0.5, 0.5, 0.5, 1.0]  # Default gray with full opacity
            
            # Extract the colors within the masked region
            pixels = image[combined_mask > 0]
            
            # If we have an alpha channel, remove it for color calculation
            if image.shape[2] == 4:
                pixels = pixels[:, :3]
                
            # Calculate mean color
            mean_color = np.mean(pixels, axis=0)
            
            if len(mean_color) == 3:  # If we got BGR
                # Convert BGR to RGB and add alpha
                return [mean_color[2]/255.0, mean_color[1]/255.0, mean_color[0]/255.0, 1.0]
            elif len(mean_color) == 4:  # If we got BGRA
                # Convert BGRA to RGBA
                return [mean_color[2]/255.0, mean_color[1]/255.0, mean_color[0]/255.0, mean_color[3]/255.0]
            
        except Exception as e:
            print(f"Color extraction error: {e}")
        
        # Default gray with full opacity
        return [0.5, 0.5, 0.5, 1.0]

    def is_fraction(self, contour, image):
        """Enhanced fraction detection with multiple approaches"""
        try:
            # Get bounding rect
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check aspect ratio (fractions are usually tall)
            aspect_ratio = h / float(w)
            if aspect_ratio < 1.2:  # More relaxed for math fractions
                return False
                
            # Get the ROI
            roi = image[y:y+h, x:x+w]
            
            # Convert to grayscale if needed
            if len(roi.shape) == 3:
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            else:
                gray_roi = roi
                
            # Threshold to binary
            _, binary = cv2.threshold(gray_roi, 1, 255, cv2.THRESH_BINARY)
            
            # Approach 1: Horizontal projection analysis
            horizontal_proj = np.sum(binary == 255, axis=1)
            mean_pixels = np.mean(horizontal_proj)
            std_pixels = np.std(horizontal_proj)
            potential_bars = np.where(horizontal_proj > mean_pixels + 2*std_pixels)[0]
            
            if len(potential_bars) > 0:
                bar_position = np.mean(potential_bars) / h
                if 0.3 < bar_position < 0.7:
                    bar_y = int(np.mean(potential_bars))
                    upper_pixels = np.count_nonzero(binary[:bar_y, :])
                    lower_pixels = np.count_nonzero(binary[bar_y:, :])
                    if upper_pixels > 10 and lower_pixels > 10:
                        return True
            
            # Approach 2: HoughLines for line detection
            edges = cv2.Canny(gray_roi, 50, 150, apertureSize=3)
            lines = cv2.HoughLines(edges, 1, np.pi/180, int(w*0.3))
            
            if lines is not None:
                for line in lines:
                    rho, theta = line[0]
                    if abs(theta - np.pi/2) < 0.3:  # More tolerant angle
                        line_y = rho / np.sin(theta) if np.sin(theta) != 0 else 0
                        if 0.3*h < line_y < 0.7*h:
                            return True
            
            # Approach 3: Morphological operations to enhance the bar
            kernel = np.ones((1, 5), np.uint8)  # Horizontal kernel
            dilated = cv2.dilate(binary, kernel, iterations=1)
            horizontal = cv2.erode(dilated, kernel, iterations=1)
            contours, _ = cv2.findContours(horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                _, _, cnt_w, cnt_h = cv2.boundingRect(cnt)
                if cnt_w > w*0.6 and cnt_h < h*0.1:  # Wide and short = likely fraction bar
                    cnt_y = cv2.boundingRect(cnt)[1]
                    if 0.3*h < cnt_y < 0.7*h:
                        return True
                        
            return False
        except Exception as e:
            print(f"Fraction detection error: {e}")
            return False

    def detect_shapes(self, image):
        """Shape detection for circles, hearts, fractions, and other polygons"""
        shapes = []
        
        try:
            if image is None or image.size == 0:
                return shapes
                
            # Convert to grayscale
            if image.shape[2] == 4:
                gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
            else:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Threshold
            _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Sort contours by area (largest first)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            for contour in contours:
                try:
                    # Skip small contours
                    if len(contour) < 5 or cv2.contourArea(contour) < 100:
                        continue
                    
                    # Get color from contour area
                    color = self.get_contour_color(contour, image)
                    
                    # Circle detection
                    (x, y), radius = cv2.minEnclosingCircle(contour)
                    area = cv2.contourArea(contour)
                    circle_area = math.pi * (radius ** 2)
                    
                    if area / circle_area > 0.85:
                        shapes.append(('circle', (x, y, radius), color))
                        continue
                    
                    # Fraction detection (try this first)
                    if self.is_fraction(contour, image):
                        shapes.append(('fraction', contour.squeeze(), color))
                        continue
                    
                    # Star detection
                    if 0.3 < area / circle_area < 0.95:
                        hull = cv2.convexHull(contour, returnPoints=False)
                        if len(hull) > 3:
                            defects = cv2.convexityDefects(contour, hull)
                            if defects is not None:
                                significant_defects = sum(1 for i in range(defects.shape[0]) 
                                    if defects[i,0,3]/256.0 > 1.0)
                                if significant_defects >= 3:
                                    shapes.append(('star', contour.squeeze(), color))
                                    continue
                    
                    # Improved Heart detection
                    if self.is_heart_shape(contour, image):
                        shapes.append(('heart', contour.squeeze(), color))
                        continue
                    
                    # Check for rectangle in true 3D mode
                    epsilon = 0.01 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    if len(approx) == 4:
                        # Calculate angles between adjacent points to check if it's a rectangle
                        rect_points = [point[0] for point in approx]
                        rect_points = np.array(rect_points, dtype=np.float32)
                        is_rectangle = True
                        
                        # Check if angles are approximately 90 degrees
                        for i in range(4):
                            p1 = rect_points[i]
                            p2 = rect_points[(i+1)%4]
                            p3 = rect_points[(i+2)%4]
                            
                            # Calculate vectors
                            v1 = p2 - p1
                            v2 = p3 - p2
                            
                            # Get angle between vectors
                            dot = np.dot(v1, v2)
                            norm = np.linalg.norm(v1) * np.linalg.norm(v2)
                            
                            if norm > 0:
                                # Calculate angle in degrees
                                angle = np.degrees(np.arccos(max(min(dot/norm, 1), -1)))
                                if abs(angle - 90) > 15:  # 15 degree tolerance
                                    is_rectangle = False
                                    break
                        
                        if is_rectangle:
                            shapes.append(('rectangle', rect_points, color))
                            continue
                    
                    # Default to polygon
                    shapes.append(('polygon', [point[0] for point in approx], color))
                    
                except Exception as e:
                    print(f"Error processing contour: {e}")
                    continue
                    
        except Exception as e:
            print(f"Shape detection error: {e}")
        
        return shapes

    def is_heart_shape(self, contour, image):
        """Improved heart shape detection with better tolerance and recognition"""
        try:
            # Basic shape checks
            if len(contour) < 10:  # Need enough points to analyze
                return False
                
            # Get basic shape properties
            area = cv2.contourArea(contour)
            if area < 200:  # Ignore very small contours
                return False
                
            perimeter = cv2.arcLength(contour, True)
            
            # Circularity check (hearts are less circular than circles)
            circularity = 4 * math.pi * area / (perimeter ** 2)
            if circularity > 0.9:  # Slightly more tolerant (was 0.85)
                return False
                
            # Get bounding rect and aspect ratio
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)
            
            # Heart typically has aspect ratio between 0.65 and 1.5 (more tolerant range)
            if aspect_ratio < 0.65 or aspect_ratio > 1.5:
                return False
                
            # Get extreme points
            extreme_top = tuple(contour[contour[:, :, 1].argmin()][0])
            extreme_bottom = tuple(contour[contour[:, :, 1].argmax()][0])
            extreme_left = tuple(contour[contour[:, :, 0].argmin()][0])
            extreme_right = tuple(contour[contour[:, :, 0].argmax()][0])
            
            # Calculate heart-specific metrics
            width = extreme_right[0] - extreme_left[0]
            height = extreme_bottom[1] - extreme_top[1]
            
            # Check center of mass (heart shape tends to have center of mass in upper half)
            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                return False
                
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            
            # Heart's center of mass should be above the geometric center
            if cy > y + (h * 0.6):  # More tolerant
                return False
            
            # Split contour into left and right parts relative to center
            left_points = [pt[0] for pt in contour if pt[0][0] < cx]
            right_points = [pt[0] for pt in contour if pt[0][0] >= cx]
            
            if len(left_points) < 3 or len(right_points) < 3:
                return False
                
            # Flip right points to compare with left (check symmetry)
            right_points_flipped = [(2*cx - x, y) for (x, y) in right_points]
            
            # Calculate similarity between left and flipped right
            similarity = self.contour_similarity(np.array(left_points), np.array(right_points_flipped))
            if similarity < 0.65:  # More tolerant (was 0.7)
                return False
                
            # IMPROVED: Check for bottom point (heart should have a point at the bottom)
            bottom_region_height = int(h * 0.3)  # Increased from 0.2
            bottom_region = [pt for pt in contour if pt[0][1] > (y + h - bottom_region_height)]
            
            if len(bottom_region) > 0:
                # Find the lowest point (should be the tip)
                tip_point = max(bottom_region, key=lambda pt: pt[0][1])
                tip_x, tip_y = tip_point[0]
                
                # The tip should be near the vertical center (more tolerant)
                if abs(tip_x - cx) > (w * 0.3):  # Increased from 0.2
                    return False
            
            # IMPROVED: Check for top dip (between lobes)
            # Use a more focused top region
            top_region_height = int(h * 0.4)  # Increased from 0.3
            top_region = [pt for pt in contour if pt[0][1] < (y + top_region_height)]
            
            if len(top_region) > 5:  # Ensure we have enough points
                # Find leftmost and rightmost points in top region
                left_top = min(top_region, key=lambda pt: pt[0][0])
                right_top = max(top_region, key=lambda pt: pt[0][0])
                
                # Check if we have points between these that are lower (indicating a dip)
                middle_x = (left_top[0][0] + right_top[0][0]) / 2
                middle_width = (right_top[0][0] - left_top[0][0]) * 0.4  # Check middle 40%
                
                middle_points = [pt for pt in top_region if 
                                abs(pt[0][0] - middle_x) < middle_width]
                
                if middle_points:
                    lowest_middle = max(middle_points, key=lambda pt: pt[0][1])
                    
                    # Check if middle dip is lower than the sides (allowing for hand-drawn imperfections)
                    left_top_y = left_top[0][1]
                    right_top_y = right_top[0][1]
                    avg_top_y = (left_top_y + right_top_y) / 2
                    
                    # If the dip isn't lower, this might not be a heart
                    if lowest_middle[0][1] < avg_top_y - 5:  # Allow slight variance
                        # No dip found, but don't immediately reject
                        # Just give this test less weight if other tests pass
                        pass
            
            # NEW: Check for convexity defects (heart shape has specific defects)
            hull = cv2.convexHull(contour, returnPoints=False)
            if len(hull) > 2:  # Need at least 3 points for convexity defects
                defects = cv2.convexityDefects(contour, hull)
                
                if defects is not None:
                    # Count significant defects - hearts typically have 1-3
                    significant_defects = 0
                    for i in range(defects.shape[0]):
                        depth = defects[i,0,3] / 256.0
                        if depth > 5:  # More tolerant threshold
                            significant_defects += 1
                    
                    # Allow a wider range of defects (more tolerant for hand-drawn hearts)
                    if significant_defects < 1 or significant_defects > 5:
                        return False
            
            # NEW: Check vertical distribution of area
            # Hearts have more area in the top than the bottom
            top_half_mask = np.zeros(image.shape[:2], dtype=np.uint8)  # Changed from self.original_image to image
            bottom_half_mask = np.zeros(image.shape[:2], dtype=np.uint8)  # Changed from self.original_image to image
            
            # Define top and bottom halves
            cv2.drawContours(top_half_mask, [contour], 0, 255, -1)
            cv2.drawContours(bottom_half_mask, [contour], 0, 255, -1)
            
            # Crop to respective halves
            mid_y = y + h/2
            top_half_mask[int(mid_y):, :] = 0  # Clear bottom part
            bottom_half_mask[:int(mid_y), :] = 0  # Clear top part
            
            # Calculate areas
            top_area = cv2.countNonZero(top_half_mask)
            bottom_area = cv2.countNonZero(bottom_half_mask)
            
            # In hearts, top area is typically larger than bottom
            if top_area <= bottom_area:
                # Not an immediate rejection, but a strong indicator
                pass  # Keep checking other criteria
                
            # If we've made it here, it's probably a heart
            return True
                
        except Exception as e:
            print(f"Heart detection error: {e}")
            return False

    def contour_similarity(self, contour1, contour2):
        """Calculate similarity between two contours using shape matching"""
        if len(contour1) < 3 or len(contour2) < 3:
            return 0.0
            
        # Resample contours to have same number of points
        target_points = min(100, min(len(contour1), len(contour2)))
        
        contour1 = self.resample_contour(contour1, target_points)
        contour2 = self.resample_contour(contour2, target_points)
        
        # Calculate similarity using Hu moments
        moments1 = cv2.moments(contour1)
        moments2 = cv2.moments(contour2)
        
        hu1 = cv2.HuMoments(moments1)
        hu2 = cv2.HuMoments(moments2)
        
        # Calculate similarity (lower is more similar)
        similarity = 0.0
        for i in range(7):
            similarity += abs(math.log(abs(hu1[i])) - math.log(abs(hu2[i])))
        
        # Normalize to 0-1 where 1 is most similar
        return 1.0 / (1.0 + similarity)

    def resample_contour(self, contour, num_points):
        """Resample contour to have exactly num_points points"""
        contour = np.squeeze(contour)
        if len(contour) == num_points:
            return contour
            
        # Calculate cumulative distance
        distances = np.zeros(len(contour))
        for i in range(1, len(contour)):
            distances[i] = distances[i-1] + np.linalg.norm(contour[i] - contour[i-1])
            
        # Create new points at equal distances
        new_points = np.zeros((num_points, 2))
        step = distances[-1] / (num_points - 1)
        
        new_points[0] = contour[0]
        current_dist = step
        original_index = 1
        
        for i in range(1, num_points-1):
            while original_index < len(contour) and distances[original_index] < current_dist:
                original_index += 1
                
            if original_index >= len(contour):
                break
                
            # Linear interpolation
            alpha = (current_dist - distances[original_index-1]) / \
                (distances[original_index] - distances[original_index-1])
            new_points[i] = contour[original_index-1] + alpha * (contour[original_index] - contour[original_index-1])
            
            current_dist += step
            
        new_points[-1] = contour[-1]
        return new_points

    def calculate_curvature(self, contour):
        """Calculate curvature at each point of the contour"""
        contour = np.squeeze(contour)
        if len(contour) < 3:
            return np.zeros(len(contour))
        
        # Calculate derivatives using central differences
        dx = np.gradient(contour[:, 0])
        dy = np.gradient(contour[:, 1])
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        
        # Calculate curvature
        curvature = (dx * ddy - dy * ddx) / ((dx**2 + dy**2)**1.5 + 1e-10)
        return curvature

    def create_fraction_mesh(self, points, height, color):
        """Create 3D fraction extrusion with enhanced geometry"""
        vertices_3d = []
        faces = []
        colors = []
        
        try:
            points = np.array(points, dtype=np.float32)
            if len(points) < 5 or np.any(np.isnan(points)) or np.any(np.isinf(points)):
                return vertices_3d, faces, colors
            
            # Create vertices
            n = len(points)
            for x, y in points:
                vertices_3d.append([x, y, 0])
                colors.append(color)
            
            for x, y in points:
                vertices_3d.append([x, y, height])
                colors.append(color)
            
            # Front face triangulation
            center_front = len(vertices_3d)
            center_x, center_y = np.mean(points[:,0]), np.mean(points[:,1])
            vertices_3d.append([center_x, center_y, 0])
            colors.append(color)
            
            for i in range(n):
                faces.append([center_front, i, (i+1)%n])
            
            # Back face triangulation
            center_back = len(vertices_3d)
            vertices_3d.append([center_x, center_y, height])
            colors.append(color)
            
            for i in range(n):
                faces.append([center_back, n+(i+1)%n, n+i])
            
            # Side faces
            for i in range(n):
                next_i = (i+1)%n
                faces.append([i, next_i, n+next_i])
                faces.append([i, n+next_i, n+i])
                
        except Exception as e:
            print(f"Fraction mesh error: {e}")
            return [], [], []
        
        return vertices_3d, faces, colors

    def create_polygon_mesh(self, vertices_2d, height, color):
        """Create a simple extruded polygon mesh or a 3D prism/box"""
        vertices_3d = []
        faces = []
        colors = []
        
        try:
            if len(vertices_2d) < 3:
                return vertices_3d, faces, colors
            
            vertices_2d = np.array(vertices_2d, dtype=np.float32)
            if np.any(np.isnan(vertices_2d)) or np.any(np.isinf(vertices_2d)):
                return vertices_3d, faces, colors
            
            n = len(vertices_2d)
            center_x, center_y = np.mean(vertices_2d[:,0]), np.mean(vertices_2d[:,1])
            
            if not self.true_3d_mode:
                # Standard extrusion
                # Front face
                for x, y in vertices_2d:
                    vertices_3d.append([x, y, 0])
                    colors.append(color)
                
                # Back face
                back_start = n
                for x, y in vertices_2d:
                    vertices_3d.append([x, y, height])
                    colors.append(color)
                
                # Front center
                center_front = len(vertices_3d)
                vertices_3d.append([center_x, center_y, 0])
                colors.append(color)
                
                for i in range(n):
                    faces.append([center_front, i, (i+1)%n])
                
                # Back center
                center_back = len(vertices_3d)
                vertices_3d.append([center_x, center_y, height])
                colors.append(color)
                
                for i in range(n):
                    faces.append([center_back, back_start+(i+1)%n, back_start+i])
                
                # Sides
                for i in range(n):
                    next_i = (i+1)%n
                    faces.append([i, next_i, back_start+next_i])
                    faces.append([i, back_start+next_i, back_start+i])
            else:
                # True 3D prism with depth - create a 3D shape based on the polygon
                # Calculate the largest dimension to normalize size
                max_dim = max(np.max(vertices_2d[:,0]) - np.min(vertices_2d[:,0]),
                              np.max(vertices_2d[:,1]) - np.min(vertices_2d[:,1]))
                depth = max_dim * 0.8 * height  # Make depth proportional to width/height and slider value
                
                # Create front face
                for x, y in vertices_2d:
                    vertices_3d.append([x, y, -depth/2])
                    colors.append(color)
                
                # Create back face
                back_start = n
                for x, y in vertices_2d:
                    vertices_3d.append([x, y, depth/2])
                    colors.append(color)
                
                # Create sides for 3D prism
                # Front face triangulation (same as before)
                center_front = len(vertices_3d)
                vertices_3d.append([center_x, center_y, -depth/2])
                colors.append(color)
                
                for i in range(n):
                    faces.append([center_front, i, (i+1)%n])
                
                # Back face triangulation (same as before)
                center_back = len(vertices_3d)
                vertices_3d.append([center_x, center_y, depth/2])
                colors.append(color)
                
                for i in range(n):
                    faces.append([center_back, back_start+(i+1)%n, back_start+i])
                
                # Connect sides
                for i in range(n):
                    next_i = (i+1)%n
                    faces.append([i, next_i, back_start+next_i])
                    faces.append([i, back_start+next_i, back_start+i])
            
        except Exception as e:
            print(f"Polygon mesh error: {e}")
            return [], [], []
        
        return vertices_3d, faces, colors

    def create_rectangle_mesh(self, vertices_2d, height, color):
        """Create a proper 3D box for rectangles"""
        vertices_3d = []
        faces = []
        colors = []
        
        try:
            if len(vertices_2d) != 4:
                return self.create_polygon_mesh(vertices_2d, height, color)
                
            # Get min/max coordinates to determine box dimensions
            x_min, y_min = np.min(vertices_2d, axis=0)
            x_max, y_max = np.max(vertices_2d, axis=0)
            
            # Calculate dimensions
            width = x_max - x_min
            height_2d = y_max - y_min
            
            # Scale depth based on slider value
            depth = height * 100  # Convert normalized height to pixels
            
            # Define the 8 vertices of a box
            # Bottom face (z = -depth/2)
            vertices_3d.append([x_min, y_min, -depth/2])  # 0: bottom left front
            vertices_3d.append([x_max, y_min, -depth/2])  # 1: bottom right front
            vertices_3d.append([x_max, y_max, -depth/2])  # 2: bottom right back
            vertices_3d.append([x_min, y_max, -depth/2])  # 3: bottom left back
            
            # Top face (z = depth/2)
            vertices_3d.append([x_min, y_min, depth/2])   # 4: top left front
            vertices_3d.append([x_max, y_min, depth/2])   # 5: top right front
            vertices_3d.append([x_max, y_max, depth/2])   # 6: top right back
            vertices_3d.append([x_min, y_max, depth/2])   # 7: top left back
            
            # Add colors for all vertices
            for _ in range(8):
                colors.append(color)
            
            # Define the 12 triangular faces (6 square faces, each made of 2 triangles)
            # Bottom face
            faces.append([0, 1, 2])
            faces.append([0, 2, 3])
            
            # Top face
            faces.append([4, 6, 5])
            faces.append([4, 7, 6])
            
            # Front face
            faces.append([0, 4, 1])
            faces.append([1, 4, 5])
            
            # Right face
            faces.append([1, 5, 2])
            faces.append([2, 5, 6])
            
            ## Right face
            faces.append([1, 5, 2])
            faces.append([2, 5, 6])
            
            # Back face
            faces.append([2, 6, 3])
            faces.append([3, 6, 7])
            
            # Left face
            faces.append([3, 7, 0])
            faces.append([0, 7, 4])
            
        except Exception as e:
            print(f"Rectangle mesh error: {e}")
            import traceback
            traceback.print_exc()
            return [], [], []
            
        return vertices_3d, faces, colors

    def create_circle_mesh(self, center, radius, height, color):
        """Create a 3D circle extrusion or a sphere"""
        vertices = []
        faces = []
        colors = []
        
        cx, cy = center
        
        if not self.true_3d_mode and not self.heart_3d_mode:
            # Standard circle extrusion
            # Front face
            front_start = 0
            for i in range(self.circle_segments):
                angle = 2 * math.pi * i / self.circle_segments
                x = cx + radius * math.cos(angle)
                y = cy + radius * math.sin(angle)
                vertices.append([x, y, 0])
                colors.append(color)
            
            # Back face
            back_start = self.circle_segments
            for i in range(self.circle_segments):
                angle = 2 * math.pi * i / self.circle_segments
                vertices.append([cx + radius * math.cos(angle), 
                                cy + radius * math.sin(angle), 
                                height * self.extrusion_strength])
                colors.append(color)
            
            # Front center
            front_center = len(vertices)
            vertices.append([cx, cy, 0])
            colors.append(color)
            
            for i in range(self.circle_segments):
                faces.append([front_center, i, (i+1)%self.circle_segments])
            
            # Back center
            back_center = len(vertices)
            vertices.append([cx, cy, height])
            colors.append(color)
            
            for i in range(self.circle_segments):
                faces.append([back_center, back_start+(i+1)%self.circle_segments, back_start+i])
            
            # Sides
            for i in range(self.circle_segments):
                next_i = (i+1)%self.circle_segments
                faces.append([i, next_i, back_start+next_i])
                faces.append([i, back_start+next_i, back_start+i])
        elif self.true_3d_mode:
            # Create a sphere
            # Generate sphere vertices and faces using UV sphere method
            for phi_idx in range(self.sphere_segments):
                phi = math.pi * phi_idx / (self.sphere_segments - 1)
                for theta_idx in range(self.sphere_segments):
                    theta = 2 * math.pi * theta_idx / self.sphere_segments
                    
                    # Sphere coordinates
                    x = cx + radius * math.sin(phi) * math.cos(theta)
                    y = cy + radius * math.sin(phi) * math.sin(theta)
                    z = radius * math.cos(phi)
                    
                    vertices.append([x, y, z])
                    colors.append(color)
            
            # Generate faces
            for phi_idx in range(self.sphere_segments - 1):
                for theta_idx in range(self.sphere_segments):
                    next_theta_idx = (theta_idx + 1) % self.sphere_segments
                    
                    # Current row indices
                    curr1 = phi_idx * self.sphere_segments + theta_idx
                    curr2 = phi_idx * self.sphere_segments + next_theta_idx
                    
                    # Next row indices
                    next1 = (phi_idx + 1) * self.sphere_segments + theta_idx
                    next2 = (phi_idx + 1) * self.sphere_segments + next_theta_idx
                    
                    # Create faces (two triangles)
                    if phi_idx > 0:  # Skip the top pole triangles
                        faces.append([curr1, curr2, next2])
                    if phi_idx < self.sphere_segments - 2:  # Skip the bottom pole triangles
                        faces.append([curr1, next2, next1])
        else:  # heart_3d_mode
            # Create a decorative heart shape
            # We'll generate a heart-shaped mesh directly using parametric equations
            horizontal_segments = 84 # More segments for even smoother shape
            vertical_segments = 64    # More vertical detail

            # Generate parametric heart
            for v_idx in range(vertical_segments + 1):
                v = v_idx / vertical_segments
                for h_idx in range(horizontal_segments): 
                    u = 2 * math.pi * h_idx / horizontal_segments
                    
                    # Use a heart-shaped function (cardioid-based)
                    # Modified version of the parametric heart formula
                    cos_u = math.cos(u)
                    sin_u = math.sin(u)
                    
                    # Create heart shape with x^2 + (y - x^(2/3))^2 = 1
                    # We adapt it to 3D using spherical coordinates
                    x = radius * (
                        16 * math.sin(u) ** 3 * math.sin(v * math.pi)
                    ) / 16
                    
                    y = radius * (
                        (13 * math.cos(u) - 5 * math.cos(2*u) - 2 * math.cos(3*u) - math.cos(4*u)) * math.sin(v * math.pi)
                    ) / 16
                    
                    z = radius * math.cos(v * math.pi) * 0.8  # Scale z-dimension
                    
                    # Offset to center position
                    vertices.append([cx + x, cy + y, z * height])
                    colors.append(color)
            
            # Generate faces connecting vertices
            for v_idx in range(vertical_segments):
                for h_idx in range(horizontal_segments):
                    # Current row indices
                    curr1 = v_idx * horizontal_segments + h_idx
                    curr2 = v_idx * horizontal_segments + (h_idx + 1) % horizontal_segments
                    
                    # Next row indices
                    next1 = (v_idx + 1) * horizontal_segments + h_idx
                    next2 = (v_idx + 1) * horizontal_segments + (h_idx + 1) % horizontal_segments
                    
                    # Create faces (two triangles)
                    faces.append([curr1, curr2, next2])
                    faces.append([curr1, next2, next1])
                        
        return vertices, faces, colors

    def create_realistic_heart_mesh(self, center, size, height, color):
        """Create a smooth, balloon-like 3D heart shape."""
        vertices = []
        faces = []
        colors = []
        
        # Extract center coordinates
        cx, cy = center
        
        # Increase segments for smoother appearance
        horizontal_segments = 84  # Increased from 64
        vertical_segments = 64    # Increased from 48
        
        # Generate parametric heart with smoother curvature
        for v_idx in range(vertical_segments + 1):
            v = v_idx / vertical_segments
            phi = v * math.pi  # Map to 0-π
            
            for h_idx in range(horizontal_segments):
                u = 2 * math.pi * h_idx / horizontal_segments
                
                # Basic trig values
                sin_phi = math.sin(phi)
                cos_phi = math.cos(phi)
                sin_u = math.sin(u)
                cos_u = math.cos(u)
                
                # Heart shape modifiers with enhanced balloon-like properties
                heart_width_factor = 1.0
                heart_depth_factor = 1.0
                
                # Create lobes and bottom point with more pronounced curvature
                if phi < math.pi / 2:  # Top half (lobes)
                    # Enhanced lobe shaping for balloon-like appearance
                    heart_width_factor = 1.0 + 0.6 * sin_u * sin_phi  # Increased from 0.5 to 0.6
                    heart_depth_factor = 0.6 + 0.5 * cos_phi  # Adjusted for rounder shape
                else:  # Bottom half
                    # Enhanced tapering for balloon-like appearance
                    heart_width_factor = 1.0 - 0.6 * (phi - math.pi / 2) / (math.pi / 2)  # Increased taper
                    heart_depth_factor = 0.6 * (1.0 - (phi - math.pi / 2) / (math.pi / 2))  # Adjusted depth
                
                # Apply additional balloon-like inflation factor
                balloon_factor = 1.2  # Adjust for desired inflation (values > 1 create more balloon-like shapes)
                
                # Calculate coordinates with all factors applied
                x = cx + size * balloon_factor * heart_width_factor * sin_phi * cos_u
                y = cy + size * balloon_factor * heart_width_factor * sin_phi * sin_u
                
                # Z calculation for depth with enhanced curve
                z = height * balloon_factor * heart_depth_factor * cos_phi
                
                vertices.append([x, y, z])
                colors.append(color)
        
        # Generate faces connecting vertices
        for phi_idx in range(vertical_segments):
            for theta_idx in range(horizontal_segments):
                # Current row indices
                curr1 = phi_idx * horizontal_segments + theta_idx
                curr2 = phi_idx * horizontal_segments + (theta_idx + 1) % horizontal_segments
                
                # Next row indices
                next1 = (phi_idx + 1) * horizontal_segments + theta_idx
                next2 = (phi_idx + 1) * horizontal_segments + (theta_idx + 1) % horizontal_segments
                
                # Create faces (two triangles)
                faces.append([curr1, curr2, next2])
                faces.append([curr1, next2, next1])
        
        return vertices, faces, colors
    
    def create_default_heart_shape(self, center, size):
        """Create a default heart shape with smooth contour"""
        cx, cy = center
        points = []
        
        # Create a heart shape using parametric equations
        for i in range(36):
            angle = 2 * math.pi * i / 36
            # Heart shape formula
            x = cx + size * (16 * math.sin(angle) ** 3) / 16
            y = cy + size * (13 * math.cos(angle) - 5 * math.cos(2*angle) - 2 * math.cos(3*angle) - math.cos(4*angle)) / 16
            points.append([x, y])
        
        return np.array(points)

    def create_star_mesh(self, points, height, color):
        """Create a 3D star/sun extrusion or a 3D star with volume"""
        vertices_3d = []
        faces = []
        colors = []
        
        try:
            if len(points) < 8:
                return vertices_3d, faces, colors
            
            points = np.array(points, dtype=np.float32)
            if np.any(np.isnan(points)) or np.any(np.isinf(points)):
                return vertices_3d, faces, colors
            
            center_x, center_y = np.mean(points[:,0]), np.mean(points[:,1])
            n = len(points)
            
            if not self.true_3d_mode and not self.heart_3d_mode:
                # Standard extrusion
                # Front vertices
                for x, y in points:
                    vertices_3d.append([x, y, 0])
                    colors.append(color)
                
                # Back vertices
                back_start = n
                for x, y in points:
                    vertices_3d.append([x, y, height * self.extrusion_strength])
                    colors.append(color)
                
                # Front center
                center_front = len(vertices_3d)
                vertices_3d.append([center_x, center_y, 0])
                colors.append(color)
                
                for i in range(n):
                    faces.append([center_front, i, (i+1)%n])
                
                # Back center
                center_back = len(vertices_3d)
                vertices_3d.append([center_x, center_y, height])
                colors.append(color)
                
                for i in range(n):
                    faces.append([center_back, back_start+(i+1)%n, back_start+i])
                
                # Sides
                for i in range(n):
                    next_i = (i+1)%n
                    faces.append([i, next_i, back_start+next_i])
                    faces.append([i, back_start+next_i, back_start+i])
            else:
                # Create a 3D star with pointed tips
                # Find max distance from center to determine star size
                distances = np.sqrt(np.sum((points - np.array([center_x, center_y]))**2, axis=1))
                max_radius = np.max(distances)
                depth = max_radius * 0.5 * height  # Make depth proportional to radius and slider value
                
                # Create front face (slightly moved in -z direction)
                for x, y in points:
                    vertices_3d.append([x, y, -depth/4])
                    colors.append(color)
                
                # Create back face (slightly moved in +z direction)
                back_start = n
                for x, y in points:
                    vertices_3d.append([x, y, depth/4])
                    colors.append(color)
                
                # Center vertices
                front_center = len(vertices_3d)
                vertices_3d.append([center_x, center_y, -depth/4])
                colors.append(color)
                
                back_center = len(vertices_3d)
                vertices_3d.append([center_x, center_y, depth/4])
                colors.append(color)
                
                # Create triangulation for front and back
                for i in range(n):
                    faces.append([front_center, i, (i+1)%n])
                    faces.append([back_center, back_start+(i+1)%n, back_start+i])
                
                # Create sides
                for i in range(n):
                    next_i = (i+1)%n
                    faces.append([i, next_i, back_start+next_i])
                    faces.append([i, back_start+next_i, back_start+i])
                
                # Add pointed tips (extend points outward from center)
                # Find local maxima in distance from center (these are the star points)
                point_indices = []
                for i in range(n):
                    prev_i = (i-1)%n
                    next_i = (i+1)%n
                    
                    if (distances[i] > distances[prev_i] and 
                        distances[i] > distances[next_i] and 
                        distances[i] > 0.8 * max_radius):
                        point_indices.append(i)
                
                # Create pointed tips at these indices
                for idx in point_indices:
                    x, y = points[idx]
                    # Vector from center to point
                    dx, dy = x - center_x, y - center_y
                    dist = np.sqrt(dx*dx + dy*dy)
                    
                    if dist > 0:
                        # Normalize and extend
                        nx, ny = dx/dist, dy/dist
                        tip_x = center_x + nx * max_radius * 1.2  # Extend beyond original
                        tip_y = center_y + ny * max_radius * 1.2
                        
                        # Add the tip vertex
                        tip_idx = len(vertices_3d)
                        vertices_3d.append([tip_x, tip_y, 0])
                        colors.append(color)
                        
                        # Create triangles from the tip to the front and back face points
                        faces.append([tip_idx, idx, (idx+1)%n])
                        faces.append([tip_idx, back_start+idx, back_start+(idx+1)%n])
                        
                        # Connect the tip to the side faces
                        faces.append([tip_idx, idx, back_start+idx])
                        faces.append([tip_idx, (idx+1)%n, back_start+(idx+1)%n])
                
        except Exception as e:
            print(f"Star mesh error: {e}")
            return [], [], []
        
        return vertices_3d, faces, colors
    
    def create_heart_mesh(self, points, height, color):
        """Create a 3D heart mesh from contour points or use a parametric heart"""
        vertices = []
        faces = []
        colors = []
        
        try:
            # Calculate center and size from points
            points = np.array(points, dtype=np.float32)
            if len(points) < 3 or np.any(np.isnan(points)) or np.any(np.isinf(points)):
                return vertices, faces, colors
                
            # Find the bounding rectangle to determine size and center
            x_min, y_min = np.min(points, axis=0)
            x_max, y_max = np.max(points, axis=0)
            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2
            size = max(x_max - x_min, y_max - y_min) / 2  # Use half the max dimension as size
            
            # If in heart 3D mode, create a parametric heart
            if self.heart_3d_mode:
                return self.create_realistic_heart_mesh((center_x, center_y), size, height, color)
                
            # Otherwise, use standard extrusion approach
            n = len(points)
            
            # Front face
            for x, y in points:
                vertices.append([x, y, 0])
                colors.append(color)
            
            # Back face
            back_start = n
            for x, y in points:
                vertices.append([x, y, height * self.extrusion_strength])
                colors.append(color)
            
            # Front center
            center_front = len(vertices)
            vertices.append([center_x, center_y, 0])
            colors.append(color)
            
            for i in range(n):
                faces.append([center_front, i, (i+1)%n])
            
            # Back center
            center_back = len(vertices)
            vertices.append([center_x, center_y, height])
            colors.append(color)
            
            for i in range(n):
                faces.append([center_back, back_start+(i+1)%n, back_start+i])
            
            # Sides
            for i in range(n):
                next_i = (i+1)%n
                faces.append([i, next_i, back_start+next_i])
                faces.append([i, back_start+next_i, back_start+i])
                
            return vertices, faces, colors
            
        except Exception as e:
            print(f"Heart mesh creation error: {e}")
            import traceback
            traceback.print_exc()
            return [], [], []

    def create_3d_mesh(self, image, shapes, height=1.0):
        """Create 3D mesh from detected shapes with optional vertices"""
        all_vertices = []
        all_faces = []
        all_colors = []
        face_offset = 0
        
        height_px = height * 100  # Convert normalized height to pixels
        
        # Create meshes for each shape
        for shape in shapes:
            shape_type, params, color = shape
            if shape_type == 'circle':
                x, y, radius = params
                vertices, faces, colors = self.create_circle_mesh(
                    (x, y), radius, height_px, color
                )
            elif shape_type == 'heart':
                vertices, faces, colors = self.create_heart_mesh(
                    params, height_px, color
                )
            elif shape_type == 'star':
                vertices, faces, colors = self.create_star_mesh(
                    params, height_px, color
                )
            elif shape_type == 'fraction':
                vertices, faces, colors = self.create_fraction_mesh(
                    params, height_px, color
                )
            elif shape_type == 'rectangle' and (self.true_3d_mode or self.heart_3d_mode):
                vertices, faces, colors = self.create_rectangle_mesh(
                    params, height_px, color
                )
            else:  # polygon
                vertices_2d = params
                vertices, faces, colors = self.create_polygon_mesh(
                    vertices_2d, height_px, color
                )
            
            # Apply inflation if enabled
            if self.inflation_enabled and vertices and faces:
                vertices, faces, colors = self.inflate_mesh(vertices, faces, colors)
            
            # Offset face indices for combined mesh
            faces = [[idx + face_offset for idx in face] for face in faces]
            face_offset += len(vertices)
            
            all_vertices.extend(vertices)
            all_faces.extend(faces)
            all_colors.extend(colors)
        
        if not all_vertices:
            return None
            
        # Convert to numpy arrays
        vertices = np.array(all_vertices, dtype=np.float32)
        faces = np.array(all_faces, dtype=np.uint32)
        colors = np.array(all_colors, dtype=np.float32)
        
        # Ensure colors are in RGBA format with proper alpha
        if colors.shape[1] == 3:  # If RGB, add alpha
            colors = np.column_stack([colors, np.ones(len(colors))])
        
        # Center and normalize the mesh
        vertices[:, 0] -= np.mean(vertices[:, 0])
        vertices[:, 1] -= np.mean(vertices[:, 1])
        vertices[:, 2] -= np.mean(vertices[:, 2])
        
        max_dim = np.max(np.ptp(vertices, axis=0))
        if max_dim > 0:
            vertices /= max_dim
            
        # Create mesh with colors
        mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            face_colors=colors  # Make sure colors are properly assigned
        )

        if self.smoothing_factor > 0:
            mesh = self.smooth_mesh(mesh, self.smoothing_factor)
        
        return mesh
    
    def smooth_mesh(self, mesh, factor):
        """Apply advanced smoothing to the mesh based on factor strength with center protection"""
        try:
            # If factor is very small, don't smooth
            if factor < 0.05:
                return mesh
                
            # Make sure the mesh is valid before smoothing
            if not mesh.is_watertight:
                print("Warning: Mesh is not watertight, using simplified smoothing")
                return self.simple_smooth_mesh(mesh, factor)
                
            # Check if mesh has any NaN or Inf values
            if np.any(np.isnan(mesh.vertices)) or np.any(np.isinf(mesh.vertices)):
                print("Warning: Mesh contains invalid coordinates, skipping smoothing")
                return mesh
                
            # Find center vertices (by finding vertices that are connected to many faces)
            # These are typically the center points we want to protect from sinking
            vertex_face_count = np.zeros(len(mesh.vertices), dtype=np.int32)
            for face in mesh.faces:
                for vertex in face:
                    vertex_face_count[vertex] += 1
                    
            # Vertices connected to many faces are likely center vertices
            # (typically twice or more connections than regular vertices)
            center_vertices = np.where(vertex_face_count > np.mean(vertex_face_count) * 1.5)[0]
            
            # Store original z-values for center vertices
            original_centers = {}
            for idx in center_vertices:
                original_centers[idx] = mesh.vertices[idx, 2]
            
            # Calculate iterations based on factor, but limit for stability
            iterations = max(1, min(5, int(factor * 10)))
                
            try:
                # Try Taubin smoothing first
                smoothed = self.taubin_smooth_mesh(mesh, iterations)
                
                # Restore center vertices' original z-heights
                for idx, z_val in original_centers.items():
                    smoothed.vertices[idx, 2] = z_val
                    
                return smoothed
            except ValueError as e:
                # If we get a dimension mismatch, fall back to simple smoothing
                print(f"Taubin smoothing failed with error: {e}")
                print("Falling back to simple smoothing")
                return self.simple_smooth_mesh_with_center_protection(mesh, factor, original_centers)
                
        except Exception as e:
            print(f"Smoothing failed: {e}")
            import traceback
            traceback.print_exc()
            return mesh
    
    def simple_smooth_mesh_with_center_protection(self, mesh, factor, center_vertices=None):
        """Apply simple vertex averaging smoothing to the mesh while preserving center height"""
        try:
            # Create a copy of the mesh
            smoothed = mesh.copy()
            vertices = smoothed.vertices.copy()
            faces = smoothed.faces
            
            # If center vertices weren't provided, try to detect them
            if center_vertices is None:
                center_vertices = {}
                vertex_face_count = np.zeros(len(vertices), dtype=np.int32)
                for face in faces:
                    for vertex in face:
                        vertex_face_count[vertex] += 1
                        
                # Find vertices connected to many faces (likely centers)
                potential_centers = np.where(vertex_face_count > np.mean(vertex_face_count) * 1.5)[0]
                
                # Store their original z-values
                for idx in potential_centers:
                    center_vertices[idx] = vertices[idx, 2]
            
            # Get neighboring vertices for each vertex
            neighbors = [[] for _ in range(len(vertices))]
            for face in faces:
                for i in range(3):
                    neighbors[face[i]].extend([face[(i+1)%3], face[(i+2)%3]])
            
            # Remove duplicates
            for i in range(len(neighbors)):
                neighbors[i] = list(set(neighbors[i]))
            
            # Apply smoothing
            strength = min(0.9, factor)  # Limit maximum smoothing
            new_vertices = vertices.copy()
            
            for i in range(len(vertices)):
                if not neighbors[i]:
                    continue
                    
                # For center vertices, only smooth X and Y, not Z
                if i in center_vertices:
                    # Calculate average position of neighbors
                    avg_pos = np.mean([vertices[n] for n in neighbors[i]], axis=0)
                    # Only move X and Y coordinates, keep original Z
                    new_vertices[i][0] = vertices[i][0] * (1 - strength) + avg_pos[0] * strength
                    new_vertices[i][1] = vertices[i][1] * (1 - strength) + avg_pos[1] * strength
                    # Keep original Z height
                    new_vertices[i][2] = center_vertices[i]
                else:
                    # For regular vertices, apply normal smoothing
                    avg_pos = np.mean([vertices[n] for n in neighbors[i]], axis=0)
                    new_vertices[i] = vertices[i] * (1 - strength) + avg_pos * strength
            
            # Update mesh vertices
            smoothed.vertices = new_vertices
            return smoothed
        except Exception as e:
            print(f"Simple smoothing failed: {e}")
            return mesh

    def simple_smooth_mesh(self, mesh, factor):
        """Apply simple vertex averaging smoothing to the mesh"""
        try:
            # Create a copy of the mesh
            smoothed = mesh.copy()
            vertices = smoothed.vertices.copy()
            faces = smoothed.faces
            
            # Get neighboring vertices for each vertex
            neighbors = [[] for _ in range(len(vertices))]
            for face in faces:
                for i in range(3):
                    neighbors[face[i]].extend([face[(i+1)%3], face[(i+2)%3]])
            
            # Remove duplicates
            for i in range(len(neighbors)):
                neighbors[i] = list(set(neighbors[i]))
            
            # Apply smoothing
            strength = min(0.9, factor)  # Limit maximum smoothing
            new_vertices = vertices.copy()
            
            for i in range(len(vertices)):
                if not neighbors[i]:
                    continue
                    
                # Calculate average position of neighbors
                avg_pos = np.mean([vertices[n] for n in neighbors[i]], axis=0)
                
                # Move vertex toward average position based on factor
                new_vertices[i] = vertices[i] * (1 - strength) + avg_pos * strength
            
            # Update mesh vertices
            smoothed.vertices = new_vertices
            return smoothed
        except Exception as e:
            print(f"Simple smoothing failed: {e}")
            return mesh


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("2D to 3D Shape Converter")
        self.setGeometry(100, 100, 1200, 800)
        self.converter = Shape3DConverter()
        self.current_mesh = None
        self.shapes = None  # Store detected shapes
        self.processed_image = None  # Store the current image
        self.smoothing_factor = 0.0
        self.inflation_distribution = 0.0  # 0.0 means uniform inflation

        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)
        
        # Left panel
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Back button at the top
        self.back_button = QPushButton("← Back to Main Menu")
        self.back_button.setStyleSheet("""
            QPushButton {
                background-color: #333333;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
                text-align: left;
            }
            QPushButton:hover {
                background-color: #444444;
            }
            QPushButton:pressed {
                background-color: #555555;
            }
        """)
        self.back_button.clicked.connect(self.go_back_to_landing_page)
        left_layout.addWidget(self.back_button)
        
        # Image display
        self.image_label = QLabel("No image selected")
        self.image_label.setFixedSize(400, 400)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("QLabel { background-color: #f0f0f0; border: 2px dashed #aaa; }")
        
        # Controls
        control_group = QGroupBox("Image Controls")
        control_layout = QVBoxLayout()
        
        self.load_button = QPushButton("Load Image")
        self.load_button.clicked.connect(self.load_image)
        
        self.remove_bg_button = QPushButton("Remove Background")
        self.remove_bg_button.setEnabled(False)
        self.remove_bg_button.clicked.connect(self.remove_background)
        
        self.convert_button = QPushButton("Convert to 3D")
        self.convert_button.setEnabled(False)
        self.convert_button.clicked.connect(self.detect_shapes_and_convert)
        
        self.export_button = QPushButton("Export 3D Model")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export_mesh)
        
        control_layout.addWidget(self.load_button)
        control_layout.addWidget(self.remove_bg_button)
        control_layout.addWidget(self.convert_button)
        control_layout.addWidget(self.export_button)
        control_group.setLayout(control_layout)
        
        # 3D Options group
        options_group = QGroupBox("3D Options")
        options_layout = QVBoxLayout()
        
        # Mode selection (horizontal layout for the checkboxes)
        mode_widget = QWidget()
        mode_layout = QHBoxLayout(mode_widget)
        mode_layout.setContentsMargins(0, 0, 0, 0)
        
        # True 3D mode checkbox
        self.true_3d_checkbox = QCheckBox("Sphere")
        self.true_3d_checkbox.setToolTip("Convert to volumetric 3D models instead of extrusions")
        self.true_3d_checkbox.stateChanged.connect(self.toggle_true_3d_mode)
        
        # Heart 3D mode checkbox
        self.heart_3d_checkbox = QCheckBox("Real Heart")
        self.heart_3d_checkbox.setToolTip("Create anatomically-inspired 3D hearts")
        self.heart_3d_checkbox.stateChanged.connect(self.toggle_heart_3d_mode)
        
        # Inflation checkbox - create it BEFORE trying to use it
        self.inflation_checkbox = QCheckBox("Inflate Shapes")
        self.inflation_checkbox.setToolTip("Create rounded, inflated versions of shapes")
        self.inflation_checkbox.stateChanged.connect(self.toggle_inflation_mode)
        
        mode_layout.addWidget(self.true_3d_checkbox)
        mode_layout.addWidget(self.heart_3d_checkbox)
        mode_layout.addWidget(self.inflation_checkbox)
        options_layout.addWidget(mode_widget)
        
        # Height control
        height_control = QWidget()
        height_layout = QVBoxLayout(height_control)
        height_layout.setContentsMargins(0, 0, 0, 0)
        
        self.height_slider = QSlider(Qt.Orientation.Horizontal)
        self.height_slider.setRange(10, 200)
        self.height_slider.setValue(50)
        self.height_label = QLabel("Extrusion Height: 0.5")
        
        height_layout.addWidget(QLabel("Extrusion Strength:"))
        height_layout.addWidget(self.height_slider)
        height_layout.addWidget(self.height_label)
        
        options_layout.addWidget(height_control)
        
        # Inflation slider control - create it BEFORE trying to use it
        inflation_control = QWidget()
        inflation_layout = QVBoxLayout(inflation_control)
        inflation_layout.setContentsMargins(0, 0, 0, 0)
        
        self.inflation_slider = QSlider(Qt.Orientation.Horizontal)
        self.inflation_slider.setRange(0, 100)
        self.inflation_slider.setValue(50)
        self.inflation_slider.setEnabled(False)  # Initially disabled
        self.inflation_label = QLabel("Inflation: 50%")
        
        inflation_layout.addWidget(QLabel("Inflation Amount:"))
        inflation_layout.addWidget(self.inflation_slider)
        inflation_layout.addWidget(self.inflation_label)
        
        options_layout.addWidget(inflation_control)
        
        # Distribution slider control
        distribution_control = QWidget()
        distribution_layout = QVBoxLayout(distribution_control)
        distribution_layout.setContentsMargins(0, 0, 0, 0)
        
        self.distribution_slider = QSlider(Qt.Orientation.Horizontal)
        self.distribution_slider.setRange(0, 100)
        self.distribution_slider.setValue(50)  # Start at middle point
        self.distribution_slider.setEnabled(False)  # Initially disabled
        self.distribution_label = QLabel("Inflation Distribution: 50%")
        
        options_layout.addWidget(distribution_control)
        options_group.setLayout(options_layout)
        
        left_layout.addWidget(self.image_label)
        left_layout.addWidget(control_group)
        left_layout.addWidget(options_group)
        left_layout.addStretch()
        
        # Right panel - 3D viewer
        self.viewer = gl.GLViewWidget()
        self.viewer.setCameraPosition(distance=3)
        grid = gl.GLGridItem()
        grid.setSize(2, 2)
        self.viewer.addItem(grid)
        
        main_layout.addWidget(left_panel, 1)
        main_layout.addWidget(self.viewer, 2)
        self.setCentralWidget(main_widget)
        
        # Connect sliders - do this AFTER creating all widgets
        self.height_slider.valueChanged.connect(self.update_height_and_model)
        self.inflation_slider.valueChanged.connect(self.update_inflation)
        self.distribution_slider.valueChanged.connect(self.update_distribution)
        self.inflation_checkbox.stateChanged.connect(lambda state: self.distribution_slider.setEnabled(state == Qt.CheckState.Checked.value))


        #self.smooth_checkbox = QCheckBox("Smooth Surface")
        #self.smooth_checkbox.setChecked(True)
        #self.smooth_checkbox.setToolTip("Apply high-quality smoothing to 3D objects")
        
        #mode_layout.addWidget(self.smooth_checkbox)
        
        # Connect to update function
        #self.smooth_checkbox.stateChanged.connect(self.toggle_smoothing)

    def go_back_to_landing_page(self):
        """Close this window and launch the landing page"""
        try:
            import os
            import sys
            import subprocess
            
            # Path to the landing page script
            landing_page_script = "landing_page.py"
            
            # Check if the landing page file exists
            if os.path.exists(landing_page_script):
                # Create a loading message
                msg = QLabel("Returning to Main Menu...")
                msg.setStyleSheet("""
                    background-color: #333333;
                    color: white;
                    padding: 10px;
                    border-radius: 5px;
                    font-weight: bold;
                """)
                msg.setFixedSize(300, 40)
                msg.setAlignment(Qt.AlignmentFlag.AlignCenter)
                
                # Position at the center of the window
                msg.setParent(self.centralWidget())
                msg.move(
                    (self.centralWidget().width() - msg.width()) // 2,
                    (self.centralWidget().height() - msg.height()) // 2
                )
                msg.show()
                QApplication.processEvents()
                
                # Start the landing page script directly using subprocess
                if sys.platform == 'win32':  # For Windows
                    subprocess.Popen([sys.executable, landing_page_script], 
                                creationflags=subprocess.CREATE_NEW_CONSOLE)
                else:  # For macOS and Linux
                    subprocess.Popen([sys.executable, landing_page_script])
                
                # Close the current window after a short delay
                QTimer.singleShot(500, self.close)
            else:
                # Show error message if landing page script not found
                QMessageBox.warning(
                    self,
                    "Error",
                    f"Landing page script '{landing_page_script}' not found!"
                )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to return to landing page: {str(e)}"
            )

    def toggle_smoothing(self, state):
        """Toggle smoothing on/off"""
        is_enabled = state == Qt.CheckState.Checked.value
        
        # Update smoothing factor based on checkbox
        self.smoothing_factor = 0.5 if is_enabled else 0.0
        self.converter.set_smoothing_factor(self.smoothing_factor)
        
        # Update the model if shapes are already detected
        if self.shapes:
            self.update_3d_model()


    def set_inflation_distribution(self, factor):
        """Set inflation distribution factor (-1.0 to 1.0)
        Negative values focus inflation at edges
        Positive values focus inflation at center
        Zero creates uniform inflation"""
        self.inflation_distribution = max(-1.0, min(1.0, factor))

    def update_distribution(self, value):
        """Update inflation distribution when slider changes"""
        # Update label
        self.distribution_label.setText(f"Inflation Distribution: {value}%")
        
        # Notify the converter
        # Values below 50 concentrate inflation at edges
        # Values above 50 concentrate inflation at center
        distribution_factor = (value - 50) / 50.0  # Range from -1.0 to 1.0
        
        # You'd need to add this attribute and method to Shape3DConverter
        self.converter.set_inflation_distribution(distribution_factor)
        
        # Update the model if shapes are already detected
        if self.shapes:
            self.update_3d_model()

    def toggle_inflation_mode(self, state):
        """Toggle inflation mode on/off"""
        is_enabled = state == Qt.CheckState.Checked.value
        self.converter.set_inflation_enabled(is_enabled)
        self.inflation_slider.setEnabled(is_enabled)
        
        # Update the model if shapes are already detected
        if self.shapes:
            self.update_3d_model()

    def update_smoothing(self, value):
        """Update smoothing factor and refresh model"""
        self.smoothing_factor = value / 100.0
        self.smoothing_label.setText(f"Edge Smoothing: {value}%")
        self.converter.set_smoothing_factor(self.smoothing_factor)
        
        if self.shapes:  # Refresh model if we have shapes
            self.update_3d_model()

    def update_inflation(self, value):
        """Update inflation factor when slider changes"""
        factor = value / 100.0
        self.inflation_label.setText(f"Inflation: {value}%")
        self.converter.set_inflation_factor(factor)
        
        # Update the model if shapes are already detected
        if self.shapes:
            self.update_3d_model()
            
    def toggle_true_3d_mode(self, state):
        """Toggle between standard extrusion and true 3D mode"""
        is_3d_mode = state == Qt.CheckState.Checked.value
        self.converter.set_true_3d_mode(is_3d_mode)
        
        # If enabling true 3D mode, uncheck heart 3D mode
        if is_3d_mode:
            self.heart_3d_checkbox.setChecked(False)
            self.height_label.setText(f"Volume: {self.height_slider.value()/100:.2f}")
        else:
            self.height_label.setText(f"Extrusion Height: {self.height_slider.value()/100:.2f}")
            
        # Update the model if shapes are already detected
        if self.shapes:
            self.update_3d_model()
            
    def toggle_heart_3d_mode(self, state):
        """Toggle between standard extrusion and heart 3D mode"""
        is_heart_mode = state == Qt.CheckState.Checked.value
        self.converter.set_heart_3d_mode(is_heart_mode)
        
        # If enabling heart 3D mode, uncheck true 3D mode
        if is_heart_mode:
            self.true_3d_checkbox.setChecked(False)
            self.height_label.setText(f"Heart Detail: {self.height_slider.value()/100:.2f}")
        else:
            self.height_label.setText(f"Extrusion Height: {self.height_slider.value()/100:.2f}")
            
        # Update the model if shapes are already detected
        if self.shapes:
            self.update_3d_model()
            
    def update_height_and_model(self, value):
        """Update the height label and the 3D model when slider changes"""
        # Update label based on current mode
        if self.true_3d_checkbox.isChecked():
            self.height_label.setText(f"Volume: {value/100:.2f}")
        elif self.heart_3d_checkbox.isChecked():
            self.height_label.setText(f"Heart Detail: {value/100:.2f}")
        else:
            self.height_label.setText(f"Extrusion Height: {value/100:.2f}")

        self.converter.set_extrusion_strength(value/100.0)
        # Update 3D model if we have shapes
        if self.shapes:
            self.update_3d_model()

    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )
        if file_name:
            self.image_path = file_name
            pixmap = QPixmap(file_name)
            self.image_label.setPixmap(pixmap.scaled(
                400, 400, Qt.AspectRatioMode.KeepAspectRatio
            ))
            self.remove_bg_button.setEnabled(True)
            self.convert_button.setEnabled(True)
            self.original_image = cv2.imread(file_name)
            self.processed_image = None
            self.shapes = None  # Reset shapes

    def remove_background(self):
        if hasattr(self, 'original_image'):
            result, _ = self.converter.remove_background(self.original_image)
            height, width, _ = result.shape
            bytes_per_line = 4 * width
            q_img = QImage(result.data, width, height, bytes_per_line, QImage.Format.Format_RGBA8888)
            pixmap = QPixmap.fromImage(q_img)
            self.image_label.setPixmap(pixmap.scaled(
                400, 400, Qt.AspectRatioMode.KeepAspectRatio
            ))
            self.processed_image = result
            self.shapes = None  # Reset shapes

    def detect_shapes_and_convert(self):
        """Detect shapes and convert to 3D"""
        try:
            image = self.processed_image if hasattr(self, 'processed_image') and self.processed_image is not None else self.original_image
            if image is None:
                QMessageBox.warning(self, "Error", "No image loaded")
                return
                
            # Detect shapes
            self.shapes = self.converter.detect_shapes(image)
            if not self.shapes:
                QMessageBox.warning(self, "Error", "No shapes detected in the image")
                return
            
            # Check if a heart shape is detected
            has_heart = any(shape[0] == 'heart' for shape in self.shapes)
            
            # If heart is detected, auto-enable inflation for balloon-like effect
            if has_heart:
                self.inflation_checkbox.setChecked(True)
                self.converter.set_inflation_enabled(True)
                self.inflation_slider.setEnabled(True)
                self.inflation_slider.setValue(80)  # Set higher inflation for hearts
                
            # Now create the 3D model
            self.update_3d_model()
            self.export_button.setEnabled(True)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Shape detection failed: {str(e)}")
            import traceback
            print(f"Error in detect_shapes_and_convert: {traceback.format_exc()}")

    def update_3d_model(self):
        """Update the 3D model using current height setting and detected shapes"""
        try:
            if not self.shapes:
                return
                
            height_factor = self.height_slider.value() / 100.0
            
            # Create 3D mesh with current settings
            self.current_mesh = self.converter.create_3d_mesh(
                self.processed_image if self.processed_image is not None else self.original_image, 
                self.shapes, 
                height_factor
            )
            
            if self.current_mesh is None:
                QMessageBox.warning(self, "Error", "Failed to create 3D mesh")
                return
                
            # Display in 3D viewer
            self.display_mesh(self.current_mesh)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"3D conversion failed: {str(e)}")
            import traceback
            print(f"Error in update_3d_model: {traceback.format_exc()}")


    def display_mesh(self, mesh):
        """Display mesh with realistic rendering and clean grid"""
        self.viewer.clear()
        
        try:
            vertices = mesh.vertices
            faces = mesh.faces
            
            # Get colors from mesh
            if hasattr(mesh.visual, 'face_colors'):
                colors = mesh.visual.face_colors
                # If colors are in 0-255 range, normalize to 0-1
                if colors.max() > 1.0:
                    colors = colors.astype(np.float32) / 255.0
            else:
                # Create default colors if none exist
                colors = np.ones((len(faces), 4)) * [0.5, 0.5, 0.5, 1.0]
            
            # Enhance colors for more realistic appearance
            enhanced_colors = colors.copy()
            # Add slight specular highlight to make it look more 3D
            enhanced_colors[:, 0:3] = np.clip(enhanced_colors[:, 0:3] * 1.2, 0, 1)
                
            # Create the mesh item with improved rendering settings
            mesh_item = gl.GLMeshItem(
                vertexes=vertices,
                faces=faces,
                faceColors=enhanced_colors,
                smooth=True,  # Enable smooth shading
                shader='shaded',  # Use shaded rendering mode for more realism
                glOptions='opaque',  # Opaque rendering for better depth perception
                drawEdges=False)  # No edges for smoother appearance
                
            self.viewer.addItem(mesh_item)
            
            # Find the minimum z value to place grid at
            z_min = np.min(vertices[:, 2])
            
            # REMOVED: Ground plane with color
            # Instead, just add a clean grid at the proper height
            
            # Add a more detailed grid for better spatial reference
            grid = gl.GLGridItem()
            # Make grid larger than the object
            x_min, y_min, _ = np.min(vertices, axis=0)
            x_max, y_max, _ = np.max(vertices, axis=0)
            grid_size = max(abs(x_max - x_min), abs(y_max - y_min)) * 3.0  # Larger grid
            grid.setSize(grid_size, grid_size)
            grid.setSpacing(grid_size/20, grid_size/20)  # More grid lines for better reference
            grid.translate(0, 0, z_min - 0.02)  # Place just below the object
            
            # Set grid to a subtle light gray (more visible but not distracting)
            grid.setColor((0.8, 0.8, 0.8, 0.7))
            
            # Add the grid to the scene
            self.viewer.addItem(grid)
            
            # Set up good lighting with optimal camera position
            self.viewer.opts['distance'] = 3.5  # Adjust camera distance
            self.viewer.opts['elevation'] = 30  # View from above
            self.viewer.opts['azimuth'] = 45    # Angle from side
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to display mesh: {str(e)}")
            print(f"Error in display_mesh: {e}")
            
        # Force the viewer to update with the new rendering settings
        self.viewer.update()

    def export_mesh(self):
        if self.current_mesh is None:
            return
            
        file_name, _ = QFileDialog.getSaveFileName(
            self, "Save 3D Model", "", "STL Files (*.stl);;OBJ Files (*.obj);;GLTF Files (*.gltf)"
        )
        
        if file_name:
            if file_name.endswith('.stl'):
                self.current_mesh.export(file_name, file_type='stl')
            elif file_name.endswith('.obj'):
                self.current_mesh.export(file_name, file_type='obj')
            elif file_name.endswith('.gltf'):
                self.current_mesh.export(file_name, file_type='gltf')
            else:
                file_name += '.stl'
                self.current_mesh.export(file_name, file_type='stl') 
                
            QMessageBox.information(self, "Success", f"Model saved to {file_name}")

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()