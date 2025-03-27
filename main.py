import sys
import cv2
import numpy as np
import trimesh
from PIL import Image
from rembg import remove
import math
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                           QPushButton, QLabel, QFileDialog, QMessageBox, QSlider, QCheckBox)
from PyQt6.QtGui import QPixmap, QImage
import pyqtgraph.opengl as gl

class Shape3DConverter:
    def __init__(self):
        self.circle_segments = 36  # Segments for circle approximation
        self.sphere_segments = 32  # Segments for sphere generation
        self.true_3d_mode = False  # Default to extrusion mode

    def set_true_3d_mode(self, enabled):
        """Enable or disable true 3D mode"""
        self.true_3d_mode = enabled

    def remove_background(self, image):
        """Remove background using rembg"""
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        output = remove(pil_img)
        output_array = np.array(output)
        bgra = cv2.cvtColor(output_array, cv2.COLOR_RGBA2BGRA)
        mask = (bgra[:, :, 3] > 0).astype(np.uint8) * 255
        return bgra, mask

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
                    
                    # Circle detection
                    (x, y), radius = cv2.minEnclosingCircle(contour)
                    area = cv2.contourArea(contour)
                    circle_area = math.pi * (radius ** 2)
                    
                    if area / circle_area > 0.85:
                        shapes.append(('circle', (x, y, radius)))
                        continue
                    
                    # Fraction detection (try this first)
                    if self.is_fraction(contour, image):
                        shapes.append(('fraction', contour.squeeze()))
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
                                    shapes.append(('star', contour.squeeze()))
                                    continue
                    
                    # Heart detection
                    extreme_top = tuple(contour[contour[:, :, 1].argmin()][0])
                    extreme_bottom = tuple(contour[contour[:, :, 1].argmax()][0])
                    width = contour[:, :, 0].max() - contour[:, :, 0].min()
                    height = extreme_bottom[1] - extreme_top[1]
                    
                    if 0.7 <= (width/height) <= 1.4:
                        moments = cv2.moments(contour)
                        if moments["m00"] != 0:
                            cx = int(moments["m10"]/moments["m00"])
                            cy = int(moments["m01"]/moments["m00"])
                            mid_x = (contour[:, :, 0].min() + contour[:, :, 0].max())/2
                            symmetry_score = abs(cx - mid_x)/width
                            
                            if symmetry_score < 0.15:
                                shapes.append(('heart', contour.squeeze()))
                                continue
                    
                    # Default to polygon
                    epsilon = 0.01 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    shapes.append(('polygon', [point[0] for point in approx]))
                    
                except Exception as e:
                    print(f"Error processing contour: {e}")
                    continue
                    
        except Exception as e:
            print(f"Shape detection error: {e}")
        
        return shapes

    def create_fraction_mesh(self, points, height, image):
        """Create 3D fraction extrusion with enhanced geometry"""
        vertices_3d = []
        faces = []
        colors = []
        
        try:
            points = np.array(points, dtype=np.float32)
            if len(points) < 5 or np.any(np.isnan(points)) or np.any(np.isinf(points)):
                return vertices_3d, faces, colors
            
            # Calculate center and color
            center_x, center_y = np.mean(points[:,0]), np.mean(points[:,1])
            color = [0.5, 0.5, 0.5]  # Default
            if 0 <= center_y < image.shape[0] and 0 <= center_x < image.shape[1]:
                color = image[int(center_y), int(center_x)][:3]/255.0
            
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

    def create_polygon_mesh(self, vertices_2d, height, image):
        """Create a simple extruded polygon mesh or a 3D prism"""
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
            color = [0.5, 0.5, 0.5]
            if 0 <= center_y < image.shape[0] and 0 <= center_x < image.shape[1]:
                color = image[int(center_y), int(center_x)][:3]/255.0
            
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
                depth = max_dim * 0.8  # Make depth proportional to width/height
                
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

    def create_circle_mesh(self, center, radius, height, image):
        """Create a 3D circle extrusion or a sphere"""
        vertices = []
        faces = []
        colors = []
        
        cx, cy = center
        color = [0.5, 0.5, 0.5]
        if 0 <= cy < image.shape[0] and 0 <= cx < image.shape[1]:
            color = image[int(cy), int(cx)][:3]/255.0
        
        if not self.true_3d_mode:
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
                                height])
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
        else:
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
                        
        return vertices, faces, colors

    def create_heart_mesh(self, points, height, image):
        """Create a 3D heart extrusion or a 3D heart shape"""
        vertices_3d = []
        faces = []
        colors = []
        
        try:
            if len(points) < 5:
                return vertices_3d, faces, colors
            
            points = np.array(points, dtype=np.float32)
            if np.any(np.isnan(points)) or np.any(np.isinf(points)):
                return vertices_3d, faces, colors
            
            center_x, center_y = np.mean(points[:,0]), np.mean(points[:,1])
            color = [0.5, 0.5, 0.5]
            if 0 <= center_y < image.shape[0] and 0 <= center_x < image.shape[1]:
                color = image[int(center_y), int(center_x)][:3]/255.0
            
            if not self.true_3d_mode:
                # Standard heart extrusion
                n = len(points)
                
                # Front vertices
                for x, y in points:
                    vertices_3d.append([x, y, 0])
                    colors.append(color)
                
                # Back vertices
                back_start = n
                for x, y in points:
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
                # Create a 3D heart with volume
                # Get heart dimensions from contour
                x_min, y_min = np.min(points, axis=0)
                x_max, y_max = np.max(points, axis=0)
                width = x_max - x_min
                height_2d = y_max - y_min
                depth = width * 0.6  # Proportional depth
                
                # Create a modified heart shape with depth
                n = len(points)
                
                # Front face (original contour)
                for x, y in points:
                    vertices_3d.append([x, y, -depth/2])
                    colors.append(color)
                
                # Back face (original contour shifted in z)
                back_start = n
                for x, y in points:
                    vertices_3d.append([x, y, depth/2])
                    colors.append(color)
                
                # Front center
                center_front = len(vertices_3d)
                vertices_3d.append([center_x, center_y, -depth/2])
                colors.append(color)
                
                for i in range(n):
                    faces.append([center_front, i, (i+1)%n])
                
                # Back center
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
                
                # Add extra vertices to create a more rounded, volumetric heart shape
                if n >= 20:  # Only if we have enough points for a smooth shape
                    # Add internal vertices to create curves
                    mid_z = 0
                    bulge = depth * 0.3  # Bulge factor
                    
                    # Top bulge vertices (for the rounded top lobes)
                    top_indices = np.where(points[:,1] < center_y)[0]
                    if len(top_indices) > 0:
                        for idx in top_indices:
                            x, y = points[idx]
                            dist_from_center = ((x - center_x)**2 + (y - center_y)**2)**0.5
                            if dist_from_center > width * 0.3:  # Only outer points
                                z_offset = bulge * (1 - abs(x - center_x) / (width * 0.5))
                                vertices_3d.append([x, y, mid_z + z_offset])
                                colors.append(color)
                                
                                # Create faces connecting to front and back
                                vid = len(vertices_3d) - 1
                                if idx > 0:
                                    faces.append([idx, idx-1, vid])
                                if idx < n-1:
                                    faces.append([idx, vid, idx+1])
                                    
                                faces.append([idx, vid, back_start+idx])
                
        except Exception as e:
            print(f"Heart mesh error: {e}")
            return [], [], []
        
        return vertices_3d, faces, colors

    def create_star_mesh(self, points, height, image):
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
            color = [1.0, 0.8, 0.0]  # Default yellow
            if 0 <= center_y < image.shape[0] and 0 <= center_x < image.shape[1]:
                color = image[int(center_y), int(center_x)][:3]/255.0
            
            n = len(points)
            
            if not self.true_3d_mode:
                # Standard extrusion
                # Front vertices
                for x, y in points:
                    vertices_3d.append([x, y, 0])
                    colors.append(color)
                
                # Back vertices
                back_start = n
                for x, y in points:
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
                # Create a 3D star with pointed tips
                # Find max distance from center to determine star size
                distances = np.sqrt(np.sum((points - np.array([center_x, center_y]))**2, axis=1))
                max_radius = np.max(distances)
                depth = max_radius * 0.5  # Make depth proportional to radius
                
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

    def create_3d_mesh(self, image, shapes, height=1.0):
        """Create 3D mesh from detected shapes"""
        all_vertices = []
        all_faces = []
        all_colors = []
        face_offset = 0
        
        height_px = height * 100  # Convert normalized height to pixels
        
        for shape_type, params in shapes:
            if shape_type == 'circle':
                x, y, radius = params
                vertices, faces, colors = self.create_circle_mesh(
                    (x, y), radius, height_px, image
                )
            elif shape_type == 'heart':
                vertices, faces, colors = self.create_heart_mesh(
                    params, height_px, image
                )
            elif shape_type == 'star':
                vertices, faces, colors = self.create_star_mesh(
                    params, height_px, image
                )
            elif shape_type == 'fraction':
                vertices, faces, colors = self.create_fraction_mesh(
                    params, height_px, image
                )
            else:  # polygon
                vertices_2d = params
                vertices, faces, colors = self.create_polygon_mesh(
                    vertices_2d, height_px, image
                )
            
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
        
        # Center and normalize the mesh
        vertices[:, 0] -= np.mean(vertices[:, 0])
        vertices[:, 1] -= np.mean(vertices[:, 1])
        vertices[:, 2] -= np.mean(vertices[:, 2])
        
        max_dim = np.max(np.ptp(vertices, axis=0))
        if max_dim > 0:
            vertices /= max_dim
            
        # Create mesh
        mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            face_colors=colors
        )
        
        return mesh

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("2D to 3D Shape Converter")
        self.setGeometry(100, 100, 1200, 800)
        self.converter = Shape3DConverter()
        self.current_mesh = None
        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)
        
        # Left panel
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Image display
        self.image_label = QLabel("No image selected")
        self.image_label.setFixedSize(400, 400)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("QLabel { background-color: #f0f0f0; border: 2px dashed #aaa; }")
        
        # Controls
        self.load_button = QPushButton("Load Image")
        self.load_button.clicked.connect(self.load_image)
        
        self.remove_bg_button = QPushButton("Remove Background")
        self.remove_bg_button.setEnabled(False)
        self.remove_bg_button.clicked.connect(self.remove_background)
        
        # True 3D mode checkbox
        self.true_3d_checkbox = QCheckBox("True 3D Mode")
        self.true_3d_checkbox.setToolTip("Convert to volumetric 3D models instead of extrusions")
        self.true_3d_checkbox.stateChanged.connect(self.toggle_true_3d_mode)
        
        self.convert_button = QPushButton("Convert to 3D")
        self.convert_button.setEnabled(False)
        self.convert_button.clicked.connect(self.convert_to_3d)
        
        self.export_button = QPushButton("Export 3D Model")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export_mesh)
        
        # Height control
        self.height_slider = QSlider(Qt.Orientation.Horizontal)
        self.height_slider.setRange(10, 200)
        self.height_slider.setValue(50)
        self.height_label = QLabel("Extrusion Height: 0.5")
        
        left_layout.addWidget(self.image_label)
        left_layout.addWidget(self.load_button)
        left_layout.addWidget(self.remove_bg_button)
        left_layout.addWidget(self.true_3d_checkbox)
        left_layout.addWidget(QLabel("Extrusion Strength:"))
        left_layout.addWidget(self.height_slider)
        left_layout.addWidget(self.height_label)
        left_layout.addWidget(self.convert_button)
        left_layout.addWidget(self.export_button)
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
        
        # Connect slider
        self.height_slider.valueChanged.connect(self.update_height_label)

    def toggle_true_3d_mode(self, state):
        """Toggle between standard extrusion and true 3D mode"""
        self.converter.set_true_3d_mode(state == Qt.CheckState.Checked.value)
        if state == Qt.CheckState.Checked.value:
            self.height_label.setText("Volume: 0.5")
        else:
            self.height_label.setText(f"Extrusion Height: {self.height_slider.value()/100:.2f}")

    def update_height_label(self, value):
        """Update the height label based on the current mode"""
        if self.true_3d_checkbox.isChecked():
            self.height_label.setText(f"Volume: {value/100:.2f}")
        else:
            self.height_label.setText(f"Extrusion Height: {value/100:.2f}")

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

    def convert_to_3d(self):
        try:
            image = self.processed_image if hasattr(self, 'processed_image') else self.original_image
            if image is None:
                QMessageBox.warning(self, "Error", "No image loaded")
                return
                
            height_factor = self.height_slider.value() / 100.0
            
            # Detect shapes
            shapes = self.converter.detect_shapes(image)
            if not shapes:
                QMessageBox.warning(self, "Error", "No shapes detected in the image")
                return
            
            # Create 3D mesh
            self.current_mesh = self.converter.create_3d_mesh(image, shapes, height_factor)
            
            if self.current_mesh is None:
                QMessageBox.warning(self, "Error", "Failed to create 3D mesh")
                return
                
            # Display in 3D viewer
            self.display_mesh(self.current_mesh)
            self.export_button.setEnabled(True)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Conversion failed: {str(e)}")
            import traceback
            print(f"Error in convert_to_3d: {traceback.format_exc()}")

    def display_mesh(self, mesh):
        self.viewer.clear()
        
        try:
            vertices = mesh.vertices
            faces = mesh.faces
            colors = mesh.visual.face_colors / 255.0
            
            # Validate data before creating the mesh item
            if (np.isnan(vertices).any() or 
                np.isinf(vertices).any() or
                np.isnan(colors).any() or
                np.isinf(colors).any()):
                QMessageBox.warning(self, "Error", "Invalid mesh data detected")
                return
                
            # Make sure face indices are within valid range
            max_vertex_idx = len(vertices) - 1
            if np.any(faces > max_vertex_idx):
                QMessageBox.warning(self, "Error", "Invalid face indices detected")
                return
                
            # Make sure colors have correct shape
            if len(colors) != len(faces):
                # Use a single color for all faces if colors don't match
                colors = np.ones((len(faces), 4)) * [0.7, 0.7, 0.7, 1.0]
            
            mesh_item = gl.GLMeshItem(
                vertexes=vertices,
                faces=faces,
                faceColors=colors,
                smooth=False,
                drawEdges=True,
                edgeColor=(0, 0, 0, 1)
            )
            
            self.viewer.addItem(mesh_item)
            grid = gl.GLGridItem()
            grid.setSize(2, 2)
            self.viewer.addItem(grid)
            self.viewer.setCameraPosition(distance=2)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to display mesh: {str(e)}")
            import traceback
            print(f"Error in display_mesh: {traceback.format_exc()}")

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