import sys
import cv2
import os
import numpy as np
import trimesh
from PIL import Image
from rembg import remove
import math
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                           QPushButton, QLabel, QFileDialog, QMessageBox, QSlider, QCheckBox,
                           QGroupBox, QSizePolicy, QComboBox, QDoubleSpinBox)
from PyQt6.QtGui import QPixmap, QImage, QCursor, QFont, QIcon  
import pyqtgraph.opengl as gl

class Shape3DConverter:
    def __init__(self):
        self.circle_segments = 72
        self.sphere_segments = 64
        self.true_3d_mode = False
        self.smoothing_factor = 0.0
        self.inflation_enabled = False
        self.inflation_factor = 0.5
        self.inflation_distribution = 0.0
        self.extrusion_strength = 1.0
        self.corner_radius = 0.0
        self.scale_factor = 1.0 
        self.unit = 'mm'
        self.pyramid_mode = False
        self.diamond_mode = False
        self.star_mode = False  

    def set_star_mode(self, enabled):
        """Enable or disable star mode for star shapes"""
        self.star_mode = enabled

    def set_diamond_mode(self, enabled):
        """Enable or disable diamond mode for quadrilaterals"""
        self.diamond_mode = enabled

    def set_scale_factor(self, factor):
        """Set the scale factor for metric conversion"""
        self.scale_factor = factor

    def set_unit(self, unit):
        """Set the current unit system"""
        self.unit = unit
        # Update scale factor based on unit
        if unit == 'mm':
            self.scale_factor = 1.0
        elif unit == 'cm':
            self.scale_factor = 10.0
        elif unit == 'm':
            self.scale_factor = 1000.0
        elif unit == 'in':
            self.scale_factor = 25.4
        elif unit == 'ft':
            self.scale_factor = 304.8

    def set_pyramid_mode(self, enabled):
        """Enable or disable pyramid mode for triangles"""
        self.pyramid_mode = enabled

    def set_corner_radius(self, radius):
        """Set the corner radius for mesh generation"""
        self.corner_radius = max(0.0, min(1.0, radius))  # Ensure value is between 0 and 1

    def set_extrusion_strength(self, strength):
        self.extrusion_strength = max(0.1, min(3.0, strength))

    def set_inflation_enabled(self, enabled):
        self.inflation_enabled = enabled
        
    def set_inflation_factor(self, factor):
        self.inflation_factor = max(0.0, min(1.0, factor))
        
    def set_inflation_distribution(self, factor):
        self.inflation_distribution = factor

    def set_smoothing_factor(self, factor):
        self.smoothing_factor = factor

    def set_true_3d_mode(self, enabled):
        self.true_3d_mode = enabled
    
        
    def set_heart_3d_mode(self, enabled):
        self.heart_3d_mode = enabled
        if enabled:
            self.true_3d_mode = False

    def remove_background(self, image):
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        output = remove(pil_img)
        output_array = np.array(output)
        bgra = cv2.cvtColor(output_array, cv2.COLOR_RGBA2BGRA)
        mask = (bgra[:, :, 3] > 0).astype(np.uint8) * 255
        return bgra, mask

    def inflate_mesh(self, vertices, faces, colors):
        if not vertices or not faces or len(vertices) < 3:
            return vertices, faces, colors
        
        try:
            vertices = np.array(vertices, dtype=np.float32)
            faces = np.array(faces, dtype=np.int32)
            
            temp_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            
            vertex_face_count = np.zeros(len(vertices), dtype=np.int32)
            for face in faces:
                for vertex in face:
                    vertex_face_count[vertex] += 1
                    
            center_indices = np.where(vertex_face_count > np.mean(vertex_face_count) * 1.5)[0]
            
            center_heights = {}
            for idx in center_indices:
                center_heights[idx] = vertices[idx, 2]
            
            mesh_size = np.max(np.ptp(vertices, axis=0))
            inflation_distance = mesh_size * 0.8 * self.inflation_factor
            
            vertex_normals = temp_mesh.vertex_normals
            
            centroid = np.mean(vertices, axis=0)
            
            vectors_from_center = vertices - centroid
            distances_from_center = np.linalg.norm(vectors_from_center, axis=1)
            max_distance = np.max(distances_from_center)
            
            normalized_distances = distances_from_center / max_distance
            
            if self.inflation_distribution > 1.0:
                power = 2.0 - self.inflation_distribution
                scale_factors = 1.0 + inflation_distance * (1.0 - normalized_distances ** power)
            elif self.inflation_distribution < 0:
                power = 2.0 + abs(self.inflation_distribution)
                scale_factors = 1.0 + inflation_distance * normalized_distances ** power
            else:
                scale_factors = 1.0 + inflation_distance * 0.5 * (1.0 + normalized_distances)
            
            inflated_vertices = centroid + vectors_from_center * scale_factors[:, np.newaxis]
            
            normal_strength = inflation_distance * 0.3
            inflated_vertices += vertex_normals * normal_strength
            
            smooth_mesh = trimesh.Trimesh(vertices=inflated_vertices, faces=faces)
            trimesh.smoothing.filter_laplacian(smooth_mesh, iterations=4)
            inflated_vertices = smooth_mesh.vertices
            
            for idx, height in center_heights.items():
                current_height = inflated_vertices[idx, 2]
                inflated_vertices[idx, 2] = height * 0.8 + current_height * 0.2
            
            return inflated_vertices.tolist(), faces.tolist(), colors
            
        except Exception as e:
            print(f"Inflation error: {e}")
            import traceback
            traceback.print_exc()
            return vertices, faces, colors
        
    def taubin_smooth_mesh(self, mesh, iterations=5):
        try:
            smoothed = mesh.copy()
            
            lamb = 0.5    # Positive smoothing factor
            mu = -0.53    # Negative smoothing factor
            
            new_vertices = np.array(smoothed.vertices, dtype=np.float64)
            adjacency = smoothed.vertex_neighbors
            
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
                
            smoothed.vertices = new_vertices
            return smoothed
            
        except Exception as e:
            print(f"Taubin smoothing failed: {e}")
            import traceback
            traceback.print_exc()
            return mesh

    def get_contour_color(self, contour, image):
        try:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            
            if image.shape[2] == 4:
                alpha_mask = image[:,:,3] > 128
                combined_mask = cv2.bitwise_and(mask, mask, mask=alpha_mask.astype(np.uint8))
            else:
                combined_mask = mask
            
            if cv2.countNonZero(combined_mask) == 0:
                return [0.5, 0.5, 0.5, 1.0]
            
            pixels = image[combined_mask > 0]
            
            if image.shape[2] == 4:
                pixels = pixels[:, :3]
                
            mean_color = np.mean(pixels, axis=0)
            
            if len(mean_color) == 3:
                return [mean_color[2]/255.0, mean_color[1]/255.0, mean_color[0]/255.0, 1.0]
            elif len(mean_color) == 4:
                return [mean_color[2]/255.0, mean_color[1]/255.0, mean_color[0]/255.0, mean_color[3]/255.0]
            
        except Exception as e:
            print(f"Color extraction error: {e}")
        
        return [0.5, 0.5, 0.5, 1.0]

    def is_fraction(self, contour, image):
        try:
            x, y, w, h = cv2.boundingRect(contour)
            
            aspect_ratio = h / float(w)
            if aspect_ratio < 1.2:
                return False
                
            roi = image[y:y+h, x:x+w]
            
            if len(roi.shape) == 3:
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            else:
                gray_roi = roi
                
            _, binary = cv2.threshold(gray_roi, 1, 255, cv2.THRESH_BINARY)
            
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
            
            edges = cv2.Canny(gray_roi, 50, 150, apertureSize=3)
            lines = cv2.HoughLines(edges, 1, np.pi/180, int(w*0.3))
            
            if lines is not None:
                for line in lines:
                    rho, theta = line[0]
                    if abs(theta - np.pi/2) < 0.3:
                        line_y = rho / np.sin(theta) if np.sin(theta) != 0 else 0
                        if 0.3*h < line_y < 0.7*h:
                            return True
            
            kernel = np.ones((1, 5), np.uint8)
            dilated = cv2.dilate(binary, kernel, iterations=1)
            horizontal = cv2.erode(dilated, kernel, iterations=1)
            contours, _ = cv2.findContours(horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                _, _, cnt_w, cnt_h = cv2.boundingRect(cnt)
                if cnt_w > w*0.6 and cnt_h < h*0.1:
                    cnt_y = cv2.boundingRect(cnt)[1]
                    if 0.3*h < cnt_y < 0.7*h:
                        return True
                        
            return False
        except Exception as e:
            print(f"Fraction detection error: {e}")
            return False

    def detect_shapes(self, image):
        shapes = []
        
        try:
            if image is None or image.size == 0:
                return shapes
                
            if image.shape[2] == 4:
                gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
            else:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
            
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            for contour in contours:
                try:
                    if len(contour) < 5 or cv2.contourArea(contour) < 100:
                        continue
                    
                    color = self.get_contour_color(contour, image)
                    
                    # First check for triangles with a more precise epsilon
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                    # Triangle detection (3 vertices)
                    if len(approx) == 3:
                        triangle_points = np.array([point[0] for point in approx])
                        
                        # Calculate angles to distinguish from other 3-point shapes
                        vectors = [
                            triangle_points[1] - triangle_points[0],
                            triangle_points[2] - triangle_points[1],
                            triangle_points[0] - triangle_points[2]
                        ]
                        
                        # Normalize vectors
                        norms = [np.linalg.norm(v) for v in vectors]
                        if all(n > 0 for n in norms):
                            vectors = [v/n for v,n in zip(vectors, norms)]
                            
                            # Calculate angles
                            angles = [
                                np.degrees(np.arccos(np.clip(-np.dot(vectors[0], vectors[2]), -1, 1))),
                                np.degrees(np.arccos(np.clip(-np.dot(vectors[1], vectors[0]), -1, 1))),
                                np.degrees(np.arccos(np.clip(-np.dot(vectors[2], vectors[1]), -1, 1)))
                            ]
                            
                            # Valid triangle should have angles summing to ~180 degrees
                            if 160 < sum(angles) < 200:
                                shapes.append(('triangle', [point[0] for point in approx], color))
                                continue
                    
                    # Get enclosing circle for shape classification
                    (x, y), radius = cv2.minEnclosingCircle(contour)
                    area = cv2.contourArea(contour)
                    circle_area = math.pi * (radius ** 2)
                    area_ratio = area / circle_area
                    
                    # Circle detection
                    if area_ratio > 0.85:
                        shapes.append(('circle', (x, y, radius), color))
                        continue
                    
                    # Fraction detection
                    if self.is_fraction(contour, image):
                        shapes.append(('fraction', contour.squeeze(), color))
                        continue
                    
                    # Star detection - improved logic
                    if 0.3 < area_ratio < 0.95:
                        hull = cv2.convexHull(contour, returnPoints=False)
                        if len(hull) > 3:
                            defects = cv2.convexityDefects(contour, hull)
                            if defects is not None:
                                significant_defects = sum(1 for i in range(defects.shape[0]) 
                                    if defects[i,0,3]/256.0 > 1.0)
                                if significant_defects >= 3:
                                    # For better star geometry, create a more uniform star from the contour
                                    M = cv2.moments(contour)
                                    if M["m00"] != 0:
                                        cx = int(M["m10"] / M["m00"])
                                        cy = int(M["m01"] / M["m00"])
                                    else:
                                        x, y, w, h = cv2.boundingRect(contour)
                                        cx, cy = x + w // 2, y + h // 2
                                    
                                    # Create a cleaner star shape for better 3D rendering
                                    shapes.append(('star', contour.squeeze(), color))
                                    continue
                    
                    # Heart detection
                    if self.is_heart_shape(contour, image):
                        shapes.append(('heart', contour.squeeze(), color))
                        continue
                    
                    # Polygon classification with relaxed epsilon
                    epsilon = 0.04 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    vertices = [point[0] for point in approx]
                    
                    # Quadrilateral classification (4 sides)
                    if len(approx) == 4:
                        rect_points = np.array(vertices, dtype=np.float32)
                        
                        # Calculate all side lengths
                        side_lengths = []
                        for i in range(4):
                            p1 = rect_points[i]
                            p2 = rect_points[(i+1)%4]
                            side_lengths.append(np.linalg.norm(p2 - p1))
                        
                        # Calculate all angles
                        angles = []
                        for i in range(4):
                            p1 = rect_points[i]
                            p2 = rect_points[(i+1)%4]
                            p3 = rect_points[(i+2)%4]
                            
                            v1 = p2 - p1
                            v2 = p3 - p2
                            
                            dot = np.dot(v1, v2)
                            norm = np.linalg.norm(v1) * np.linalg.norm(v2)
                            if norm > 0:
                                angle = np.degrees(np.arccos(max(min(dot/norm, 1), -1)))
                                angles.append(angle)
                        
                        # Check for rectangle (all angles ~90 degrees)
                        if all(85 < angle < 95 for angle in angles):
                            shapes.append(('rectangle', rect_points, color))
                            continue
                        
                        # Check for diamond/rhombus (all sides equal within 15% tolerance)
                        side_ratio = np.std(side_lengths)/np.mean(side_lengths)
                        if side_ratio < 0.15:  # All sides roughly equal
                            shapes.append(('diamond', rect_points, color))
                            continue
                        
                        # Default quadrilateral - will be converted to diamond if diamond_mode is on
                        shapes.append(('quadrilateral', rect_points, color))
                        continue
                    
                    # Default polygon classification
                    shapes.append(('polygon', vertices, color))
                    
                except Exception as e:
                    print(f"Error processing contour: {e}")
                    continue
                    
        except Exception as e:
            print(f"Shape detection error: {e}")
        
        return shapes
    
    def is_heart_shape(self, contour, image):
        try:
            if len(contour) < 10:
                return False
                
            area = cv2.contourArea(contour)
            if area < 200:
                return False
                
            perimeter = cv2.arcLength(contour, True)
            
            circularity = 4 * math.pi * area / (perimeter ** 2)
            if circularity > 0.9:
                return False
                
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)
            
            if aspect_ratio < 0.65 or aspect_ratio > 1.5:
                return False
                
            extreme_top = tuple(contour[contour[:, :, 1].argmin()][0])
            extreme_bottom = tuple(contour[contour[:, :, 1].argmax()][0])
            extreme_left = tuple(contour[contour[:, :, 0].argmin()][0])
            extreme_right = tuple(contour[contour[:, :, 0].argmax()][0])
            
            width = extreme_right[0] - extreme_left[0]
            height = extreme_bottom[1] - extreme_top[1]
            
            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                return False
                
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            
            if cy > y + (h * 0.6):
                return False
            
            left_points = [pt[0] for pt in contour if pt[0][0] < cx]
            right_points = [pt[0] for pt in contour if pt[0][0] >= cx]
            
            if len(left_points) < 3 or len(right_points) < 3:
                return False
                
            right_points_flipped = [(2*cx - x, y) for (x, y) in right_points]
            
            similarity = self.contour_similarity(np.array(left_points), np.array(right_points_flipped))
            if similarity < 0.65:
                return False
                
            bottom_region_height = int(h * 0.3)
            bottom_region = [pt for pt in contour if pt[0][1] > (y + h - bottom_region_height)]
            
            if len(bottom_region) > 0:
                tip_point = max(bottom_region, key=lambda pt: pt[0][1])
                tip_x, tip_y = tip_point[0]
                
                if abs(tip_x - cx) > (w * 0.3):
                    return False
            
            top_region_height = int(h * 0.4)
            top_region = [pt for pt in contour if pt[0][1] < (y + top_region_height)]
            
            if len(top_region) > 5:
                left_top = min(top_region, key=lambda pt: pt[0][0])
                right_top = max(top_region, key=lambda pt: pt[0][0])
                
                middle_x = (left_top[0][0] + right_top[0][0]) / 2
                middle_width = (right_top[0][0] - left_top[0][0]) * 0.4
                
                middle_points = [pt for pt in top_region if 
                                abs(pt[0][0] - middle_x) < middle_width]
                
                if middle_points:
                    lowest_middle = max(middle_points, key=lambda pt: pt[0][1])
                    
                    left_top_y = left_top[0][1]
                    right_top_y = right_top[0][1]
                    avg_top_y = (left_top_y + right_top_y) / 2
            
            hull = cv2.convexHull(contour, returnPoints=False)
            if len(hull) > 2:
                defects = cv2.convexityDefects(contour, hull)
                
                if defects is not None:
                    significant_defects = 0
                    for i in range(defects.shape[0]):
                        depth = defects[i,0,3] / 256.0
                        if depth > 5:
                            significant_defects += 1
                    
                    if significant_defects < 1 or significant_defects > 5:
                        return False
            
            top_half_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            bottom_half_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            
            cv2.drawContours(top_half_mask, [contour], 0, 255, -1)
            cv2.drawContours(bottom_half_mask, [contour], 0, 255, -1)
            
            mid_y = y + h/2
            top_half_mask[int(mid_y):, :] = 0
            bottom_half_mask[:int(mid_y), :] = 0
            
            top_area = cv2.countNonZero(top_half_mask)
            bottom_area = cv2.countNonZero(bottom_half_mask)
                
            return True
                
        except Exception as e:
            print(f"Heart detection error: {e}")
            return False

    def contour_similarity(self, contour1, contour2):
        if len(contour1) < 3 or len(contour2) < 3:
            return 0.0
            
        target_points = min(100, min(len(contour1), len(contour2)))
        
        contour1 = self.resample_contour(contour1, target_points)
        contour2 = self.resample_contour(contour2, target_points)
        
        moments1 = cv2.moments(contour1)
        moments2 = cv2.moments(contour2)
        
        hu1 = cv2.HuMoments(moments1)
        hu2 = cv2.HuMoments(moments2)
        
        similarity = 0.0
        for i in range(7):
            similarity += abs(math.log(abs(hu1[i])) - math.log(abs(hu2[i])))
        
        return 1.0 / (1.0 + similarity)

    def resample_contour(self, contour, num_points):
        contour = np.squeeze(contour)
        if len(contour) == num_points:
            return contour
            
        distances = np.zeros(len(contour))
        for i in range(1, len(contour)):
            distances[i] = distances[i-1] + np.linalg.norm(contour[i] - contour[i-1])
            
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
                
            alpha = (current_dist - distances[original_index-1]) / \
                (distances[original_index] - distances[original_index-1])
            new_points[i] = contour[original_index-1] + alpha * (contour[original_index] - contour[original_index-1])
            
            current_dist += step
            
        new_points[-1] = contour[-1]
        return new_points
    
    
    def create_diamond_mesh(self, vertices_2d, height, color):
        """Create a 3D diamond mesh from quadrilateral base"""
        vertices_3d = []
        faces = []
        colors = []
        
        try:
            if len(vertices_2d) != 4:
                return [], [], []
            
            vertices_2d = np.array(vertices_2d, dtype=np.float32)
            if np.any(np.isnan(vertices_2d)) or np.any(np.isinf(vertices_2d)):
                return [], [], []
            
            # Calculate center and size
            center_x, center_y = np.mean(vertices_2d[:,0]), np.mean(vertices_2d[:,1])
            max_dim = max(np.max(vertices_2d[:,0]) - np.min(vertices_2d[:,0]),
                        np.max(vertices_2d[:,1]) - np.min(vertices_2d[:,1]))
            
            # Create base vertices (at z=0)
            for x, y in vertices_2d:
                vertices_3d.append([x, y, 0])
                colors.append(color)
            
            # Add apex vertices (top and bottom points)
            apex_top_idx = len(vertices_3d)
            top_height = height * 1.5  # Taller top for diamond shape
            vertices_3d.append([center_x, center_y, top_height])
            colors.append(color)
            
            apex_bottom_idx = len(vertices_3d)
            bottom_height = height * 1.0  # Bottom point
            vertices_3d.append([center_x, center_y, -bottom_height])
            colors.append(color)
            
            # Create quadrilateral base face
            faces.append([0, 1, 2, 3])
            
            # Create side faces connecting base to apexes
            for i in range(4):
                next_i = (i + 1) % 4
                # Top faces
                faces.append([i, next_i, apex_top_idx])
                # Bottom faces
                faces.append([i, apex_bottom_idx, next_i])
            
            # Add center point for better shading
            center_idx = len(vertices_3d)
            vertices_3d.append([center_x, center_y, 0])
            colors.append(color)
            
            # Add middle faces connecting to center
            for i in range(4):
                next_i = (i + 1) % 4
                faces.append([i, next_i, center_idx])
            
        except Exception as e:
            print(f"Diamond mesh error: {e}")
            import traceback
            traceback.print_exc()
            return [], [], []
        
        return vertices_3d, faces, colors

    def create_triangle_mesh(self, vertices_2d, height, color):
        """Create a regular extruded triangle mesh"""
        return self.create_polygon_mesh(vertices_2d, height, color)


    def create_fraction_mesh(self, points, height, color):
        vertices_3d = []
        faces = []
        colors = []
        
        try:
            points = np.array(points, dtype=np.float32)
            if len(points) < 5 or np.any(np.isnan(points)) or np.any(np.isinf(points)):
                return vertices_3d, faces, colors
            
            n = len(points)
            for x, y in points:
                vertices_3d.append([x, y, 0])
                colors.append(color)
            
            for x, y in points:
                vertices_3d.append([x, y, height])
                colors.append(color)
            
            center_front = len(vertices_3d)
            center_x, center_y = np.mean(points[:,0]), np.mean(points[:,1])
            vertices_3d.append([center_x, center_y, 0])
            colors.append(color)
            
            for i in range(n):
                faces.append([center_front, i, (i+1)%n])
            
            center_back = len(vertices_3d)
            vertices_3d.append([center_x, center_y, height])
            colors.append(color)
            
            for i in range(n):
                faces.append([center_back, n+(i+1)%n, n+i])
            
            for i in range(n):
                next_i = (i+1)%n
                faces.append([i, next_i, n+next_i])
                faces.append([i, n+next_i, n+i])
                
        except Exception as e:
            print(f"Fraction mesh error: {e}")
            return [], [], []
        
        return vertices_3d, faces, colors

    def create_polygon_mesh(self, vertices_2d, height, color):
        vertices_3d = []
        faces = []
        colors = []
        
        try:
            if len(vertices_2d) < 3:
                return vertices_3d, faces, colors
            
            vertices_2d = np.array(vertices_2d, dtype=np.float32)
            if np.any(np.isnan(vertices_2d)) or np.any(np.isinf(vertices_2d)):
                return vertices_3d, faces, colors
            
            # Apply corner rounding if corner_radius is > 0
            if self.corner_radius > 0 and len(vertices_2d) > 2:
                vertices_2d = self._round_polygon_corners(vertices_2d, self.corner_radius)
            
            n = len(vertices_2d)
            center_x, center_y = np.mean(vertices_2d[:,0]), np.mean(vertices_2d[:,1])
            
            if not self.true_3d_mode:
                for x, y in vertices_2d:
                    vertices_3d.append([x, y, 0])
                    colors.append(color)
                
                back_start = n
                for x, y in vertices_2d:
                    vertices_3d.append([x, y, height * self.extrusion_strength])
                    colors.append(color)
                
                center_front = len(vertices_3d)
                vertices_3d.append([center_x, center_y, 0])
                colors.append(color)
                
                for i in range(n):
                    faces.append([center_front, i, (i+1)%n])
                
                center_back = len(vertices_3d)
                vertices_3d.append([center_x, center_y, height * self.extrusion_strength])
                colors.append(color)
                
                for i in range(n):
                    faces.append([center_back, back_start+(i+1)%n, back_start+i])
                
                for i in range(n):
                    next_i = (i+1)%n
                    faces.append([i, next_i, back_start+next_i])
                    faces.append([i, back_start+next_i, back_start+i])
            else:
                max_dim = max(np.max(vertices_2d[:,0]) - np.min(vertices_2d[:,0]),
                            np.max(vertices_2d[:,1]) - np.min(vertices_2d[:,1]))
                depth = max_dim * 0.8 * height
                
                for x, y in vertices_2d:
                    vertices_3d.append([x, y, -depth/2])
                    colors.append(color)
                
                back_start = n
                for x, y in vertices_2d:
                    vertices_3d.append([x, y, depth/2])
                    colors.append(color)
                
                center_front = len(vertices_3d)
                vertices_3d.append([center_x, center_y, -depth/2])
                colors.append(color)
                
                for i in range(n):
                    faces.append([center_front, i, (i+1)%n])
                
                center_back = len(vertices_3d)
                vertices_3d.append([center_x, center_y, depth/2])
                colors.append(color)
                
                for i in range(n):
                    faces.append([center_back, back_start+(i+1)%n, back_start+i])
                
                for i in range(n):
                    next_i = (i+1)%n
                    faces.append([i, next_i, back_start+next_i])
                    faces.append([i, back_start+next_i, back_start+i])
            
        except Exception as e:
            print(f"Polygon mesh error: {e}")
            return [], [], []
        
        return vertices_3d, faces, colors
    
    def _round_polygon_corners(self, vertices, radius_factor):
        """
        Round the corners of a polygon using Bezier curve interpolation
        
        :param vertices: Original polygon vertices
        :param radius_factor: Smoothing factor (0-1)
        :return: New vertices with rounded corners
        """
        import numpy as np
        
        def interpolate_point(p1, p2, t):
            """Linear interpolation between two points"""
            return p1 * (1 - t) + p2 * t
        
        def bezier_quadratic(p0, p1, p2, t):
            """Quadratic Bezier curve interpolation"""
            a = interpolate_point(p0, p1, t)
            b = interpolate_point(p1, p2, t)
            return interpolate_point(a, b, t)
        
        rounded_vertices = []
        num_vertices = len(vertices)
        
        # Number of interpolation points for each rounded corner
        corner_segments = max(3, int(10 * radius_factor))
        
        for i in range(num_vertices):
            # Current vertex and neighboring vertices
            current = vertices[i]
            prev = vertices[(i-1)%num_vertices]
            next = vertices[(i+1)%num_vertices]
            
            # Calculate vectors
            vec1 = prev - current
            vec2 = next - current
            
            # Normalize vectors
            len1 = np.linalg.norm(vec1)
            len2 = np.linalg.norm(vec2)
            
            if len1 > 0 and len2 > 0:
                vec1 /= len1
                vec2 /= len2
                
                # Minimum corner radius to prevent overlap
                min_side_length = min(len1, len2)
                max_radius = min_side_length * 0.4
                corner_radius = radius_factor * max_radius
                
                # Control point (vertex of the original angle)
                control_point = current
                
                # Start and end points of the curve
                start_point = current + vec1 * corner_radius
                end_point = current + vec2 * corner_radius
                
                # Generate interpolated points for the rounded corner
                for j in range(corner_segments):
                    t = j / (corner_segments - 1)
                    rounded_point = bezier_quadratic(start_point, control_point, end_point, t)
                    rounded_vertices.append(rounded_point)
        
        return np.array(rounded_vertices)

    def create_rectangle_mesh(self, vertices_2d, height, color):
        vertices_3d = []
        faces = []
        colors = []
        
        try:
            # Apply corner rounding if corner_radius is > 0
            if self.corner_radius > 0:
                vertices_2d = self._round_polygon_corners(vertices_2d, self.corner_radius)

            if len(vertices_2d) != 4:
                return self.create_polygon_mesh(vertices_2d, height, color)
                
            x_min, y_min = np.min(vertices_2d, axis=0)
            x_max, y_max = np.max(vertices_2d, axis=0)
            
            depth = height * 100
            
            vertices_3d.append([x_min, y_min, -depth/2])
            vertices_3d.append([x_max, y_min, -depth/2])
            vertices_3d.append([x_max, y_max, -depth/2])
            vertices_3d.append([x_min, y_max, -depth/2])
            
            vertices_3d.append([x_min, y_min, depth/2])
            vertices_3d.append([x_max, y_min, depth/2])
            vertices_3d.append([x_max, y_max, depth/2])
            vertices_3d.append([x_min, y_max, depth/2])
            
            for _ in range(8):
                colors.append(color)
            
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
            
            # Back face
            faces.append([2, 6, 3])
            faces.append([3, 6, 7])
            
            # Left face
            faces.append([3, 7, 0])
            faces.append([0, 7, 4])
            
        except Exception as e:
            print(f"Rectangle mesh error: {e}")
            return [], [], []
            
        return vertices_3d, faces, colors
    
    def create_volumetric_star_mesh(self, points, height, color):
        """
        Create a star with an extruding center point, like the pyramid mode for triangles
        
        Args:
            points: The points from the detected contour (used only for position/size)
            height: The height/thickness of the star
            color: The color for the star
            
        Returns:
            vertices, faces, colors for the star
        """
        vertices = []
        faces = []
        colors = []
        
        try:
            # Calculate center from input points
            points = np.array(points, dtype=np.float32)
            center_x, center_y = np.mean(points[:,0]), np.mean(points[:,1])
            
            # Calculate max radius from points (for sizing)
            distances = np.sqrt((points[:,0] - center_x)**2 + (points[:,1] - center_y)**2)
            outer_radius = np.max(distances)
            inner_radius = outer_radius * 0.4  # Standard ratio for stars
            
            # Number of points for the star
            num_points = 5
            
            # Apply extrusion_strength to height for the center point
            center_height = height * self.extrusion_strength
            
            # Create star perimeter points (all at z=0, like the pyramid base)
            perimeter_start = len(vertices)
            for i in range(num_points * 2):
                angle = 2 * np.pi * i / (num_points * 2)
                radius = inner_radius if i % 2 else outer_radius
                
                x = center_x + radius * np.cos(angle - np.pi/2)
                y = center_y + radius * np.sin(angle - np.pi/2)
                
                # All perimeter points at z=0 (flat base)
                vertices.append([x, y, 0])
                colors.append(color)
            
            # Add top center apex point
            top_apex_idx = len(vertices)
            vertices.append([center_x, center_y, center_height])  # Top center point
            colors.append(color)
            
            # Add bottom center apex point
            bottom_apex_idx = len(vertices)
            vertices.append([center_x, center_y, -center_height])  # Bottom center point
            colors.append(color)
            
            # Create top faces connecting perimeter to top apex
            for i in range(num_points * 2):
                next_i = (i + 1) % (num_points * 2)
                # Connect adjacent perimeter points to top apex
                faces.append([perimeter_start + i, perimeter_start + next_i, top_apex_idx])
            
            # Create bottom faces connecting perimeter to bottom apex
            for i in range(num_points * 2):
                next_i = (i + 1) % (num_points * 2)
                # Connect adjacent perimeter points to bottom apex (reverse winding)
                faces.append([perimeter_start + next_i, perimeter_start + i, bottom_apex_idx])
            
            return vertices, faces, colors
                    
        except Exception as e:
            print(f"Star mesh error: {e}")
            import traceback
            traceback.print_exc()
            return [], [], []

    def create_pyramid_mesh(self, vertices_2d, height, color):
        """
        Create a 3D pyramid mesh from a triangle base
        
        Args:
            vertices_2d (list): List of 3 points defining the base triangle
            height (float): Height of the pyramid
            color (list): RGBA color for the pyramid
        
        Returns:
            tuple: (vertices, faces, colors) for the pyramid
        """
        vertices_3d = []
        faces = []
        colors = []
        
        try:
            if len(vertices_2d) != 3:
                return [], [], []
            
            vertices_2d = np.array(vertices_2d, dtype=np.float32)
            if np.any(np.isnan(vertices_2d)) or np.any(np.isinf(vertices_2d)):
                return [], [], []
            
            # Calculate center of the base triangle
            center_x, center_y = np.mean(vertices_2d[:,0]), np.mean(vertices_2d[:,1])
            
            # Create base triangle vertices (at z=0)
            for x, y in vertices_2d:
                vertices_3d.append([x, y, 0])
                colors.append(color)
            
            # Add apex vertex
            apex_idx = len(vertices_3d)
            vertices_3d.append([center_x, center_y, height])
            colors.append(color)
            
            # Create triangular base face
            faces.append([0, 1, 2])
            
            # Create side faces connecting base to apex
            faces.append([0, 1, apex_idx])
            faces.append([1, 2, apex_idx])
            faces.append([2, 0, apex_idx])
            
        except Exception as e:
            print(f"Pyramid mesh error: {e}")
            import traceback
            traceback.print_exc()
            return [], [], []
        
        return vertices_3d, faces, colors
    
    

    def create_circle_mesh(self, center, radius, height, color):
        vertices = []
        faces = []
        colors = []
        
        cx, cy = center
        
        if not self.true_3d_mode:  # Removed heart_3d_mode check
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
            vertices.append([cx, cy, height * self.extrusion_strength])  # Apply extrusion_strength here too
            colors.append(color)
            
            for i in range(self.circle_segments):
                faces.append([back_center, back_start+(i+1)%self.circle_segments, back_start+i])
            
            # Sides
            for i in range(self.circle_segments):
                next_i = (i+1)%self.circle_segments
                faces.append([i, next_i, back_start+next_i])
                faces.append([i, back_start+next_i, back_start+i])
        else:  # true_3d_mode
            # Create a sphere
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

    def create_realistic_heart_mesh(self, center, size, height, color):
        vertices = []
        faces = []
        colors = []
        
        cx, cy = center
        horizontal_segments = 84
        vertical_segments = 64
        
        for v_idx in range(vertical_segments + 1):
            v = v_idx / vertical_segments
            phi = v * math.pi
            
            for h_idx in range(horizontal_segments):
                u = 2 * math.pi * h_idx / horizontal_segments
                
                sin_phi = math.sin(phi)
                cos_phi = math.cos(phi)
                sin_u = math.sin(u)
                cos_u = math.cos(u)
                
                heart_width_factor = 1.0
                heart_depth_factor = 1.0
                
                if phi < math.pi / 2:
                    heart_width_factor = 1.0 + 0.6 * sin_u * sin_phi
                    heart_depth_factor = 0.6 + 0.5 * cos_phi
                else:
                    heart_width_factor = 1.0 - 0.6 * (phi - math.pi / 2) / (math.pi / 2)
                    heart_depth_factor = 0.6 * (1.0 - (phi - math.pi / 2) / (math.pi / 2))
                
                balloon_factor = 1.2
                
                x = cx + size * balloon_factor * heart_width_factor * sin_phi * cos_u
                y = cy + size * balloon_factor * heart_width_factor * sin_phi * sin_u
                z = height * balloon_factor * heart_depth_factor * cos_phi
                
                vertices.append([x, y, z])
                colors.append(color)
        
        for phi_idx in range(vertical_segments):
            for theta_idx in range(horizontal_segments):
                curr1 = phi_idx * horizontal_segments + theta_idx
                curr2 = phi_idx * horizontal_segments + (theta_idx + 1) % horizontal_segments
                
                next1 = (phi_idx + 1) * horizontal_segments + theta_idx
                next2 = (phi_idx + 1) * horizontal_segments + (theta_idx + 1) % horizontal_segments
                
                faces.append([curr1, curr2, next2])
                faces.append([curr1, next2, next1])
        
        return vertices, faces, colors
    

    def create_star_mesh(self, points, height, color):
        """Create a 3D star mesh with proper geometry"""
        vertices_3d = []
        faces = []
        colors = []
        
        try:
            # Get center of the star from input points
            points = np.array(points, dtype=np.float32)
            center_x, center_y = np.mean(points[:,0]), np.mean(points[:,1])
            
            # Calculate maximum radius from the points
            distances = np.sqrt((points[:,0] - center_x)**2 + (points[:,1] - center_y)**2)
            outer_radius = np.max(distances)
            inner_radius = outer_radius * 0.4  # Standard ratio for star points
            
            # Number of points for a standard star
            num_points = 5
            
            # Create vertices for the star
            vertices = []
            
            # Center point (at z=0)
            vertices.append([center_x, center_y, 0])
            
            # Create points around the star (on XY plane)
            for i in range(num_points * 2):
                angle = 2 * np.pi * i / (num_points * 2)
                radius = inner_radius if i % 2 else outer_radius
                
                x = center_x + radius * np.cos(angle - np.pi/2)  # -pi/2 to start at top
                y = center_y + radius * np.sin(angle - np.pi/2)
                
                vertices.append([x, y, 0])
            
            # Create different star types based on true_3d_mode
            if self.true_3d_mode:
                # Add top and bottom apex points for true 3D star
                front_apex_idx = len(vertices)
                vertices.append([center_x, center_y, height])  # Top apex
                back_apex_idx = len(vertices)
                vertices.append([center_x, center_y, -height/2])  # Bottom apex
                
                # Create faces for 3D star
                # Top faces (connecting to top apex)
                for i in range(num_points * 2):
                    next_i = (i + 1) % (num_points * 2)
                    faces.append([i + 1, next_i + 1, front_apex_idx])
                
                # Bottom faces (connecting to bottom apex)
                for i in range(num_points * 2):
                    next_i = (i + 1) % (num_points * 2)
                    faces.append([next_i + 1, i + 1, back_apex_idx])
                
                # Base faces (connecting to center)
                for i in range(num_points * 2):
                    next_i = (i + 1) % (num_points * 2)
                    faces.append([0, i + 1, next_i + 1])
            else:
                # For extrusion, create back vertices
                back_start = len(vertices)
                for i in range(num_points * 2):
                    angle = 2 * np.pi * i / (num_points * 2)
                    radius = inner_radius if i % 2 else outer_radius
                    
                    x = center_x + radius * np.cos(angle - np.pi/2)
                    y = center_y + radius * np.sin(angle - np.pi/2)
                    
                    vertices.append([x, y, height * self.extrusion_strength])
                
                # Add center vertices (front and back)
                front_center = len(vertices)
                vertices.append([center_x, center_y, 0])
                back_center = len(vertices)
                vertices.append([center_x, center_y, height * self.extrusion_strength])
                
                # Front faces (connecting to front center)
                for i in range(num_points * 2):
                    next_i = (i + 1) % (num_points * 2)
                    faces.append([front_center, i + 1, next_i + 1])
                
                # Back faces (connecting to back center)
                for i in range(num_points * 2):
                    next_i = (i + 1) % (num_points * 2)
                    # Note reversed winding order for correct normals
                    faces.append([back_center, back_start + next_i, back_start + i])
                
                # Side faces
                for i in range(num_points * 2):
                    next_i = (i + 1) % (num_points * 2)
                    # First triangle
                    faces.append([i + 1, next_i + 1, back_start + next_i])
                    # Second triangle
                    faces.append([i + 1, back_start + next_i, back_start + i])
            
            # Create colors - use the input color for all vertices
            for _ in range(len(vertices)):
                colors.append(color)
            
            return vertices, faces, colors
                
        except Exception as e:
            print(f"Star mesh error: {e}")
            import traceback
            traceback.print_exc()
            return [], [], []
    
    def create_heart_mesh(self, points, height, color):
        vertices = []
        faces = []
        colors = []
        
        try:
            points = np.array(points, dtype=np.float32)
            if len(points) < 3 or np.any(np.isnan(points)) or np.any(np.isinf(points)):
                return vertices, faces, colors
                
            x_min, y_min = np.min(points, axis=0)
            x_max, y_max = np.max(points, axis=0)
            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2
            
            n = len(points)
            
            for x, y in points:
                vertices.append([x, y, 0])
                colors.append(color)
            
            back_start = n
            for x, y in points:
                vertices.append([x, y, height * self.extrusion_strength])
                colors.append(color)
            
            center_front = len(vertices)
            vertices.append([center_x, center_y, 0])
            colors.append(color)
            
            for i in range(n):
                faces.append([center_front, i, (i+1)%n])
            
            center_back = len(vertices)
            vertices.append([center_x, center_y, height * self.extrusion_strength])  # Apply extrusion_strength here too
            colors.append(color)
            
            for i in range(n):
                faces.append([center_back, back_start+(i+1)%n, back_start+i])
            
            for i in range(n):
                next_i = (i+1)%n
                faces.append([i, next_i, back_start+next_i])
                faces.append([i, back_start+next_i, back_start+i])
                
            return vertices, faces, colors
            
        except Exception as e:
            print(f"Heart mesh creation error: {e}")
            return [], [], []

    def create_3d_mesh(self, image, shapes, height=1.0):
        all_vertices = []
        all_faces = []
        all_colors = []
        face_offset = 0
        
        height_px = height * 100
        
        for shape in shapes:
            shape_type, params, color = shape
            print(f"Processing shape: {shape_type}")  # Debug output

            if shape_type == 'star':
                print(f"Creating star with star_mode={self.star_mode}, true_3d_mode={self.true_3d_mode}")
                
                # IMPORTANT FIX: Always use the volumetric implementation for stars when star_mode is enabled
                # This ensures we create a proper 3D star with separate front and back faces
                if self.star_mode:
                    # Use volumetric star (THIS IS THE KEY CHANGE)
                    print("Using volumetric star implementation")
                    vertices, faces, colors = self.create_volumetric_star_mesh(
                        params, height_px, color
                    )
                else:
                    # Fall back to polygon mesh when star mode is off
                    vertices, faces, colors = self.create_polygon_mesh(
                        params, height_px, color
                    )
                    
            elif shape_type == 'circle':
                x, y, radius = params
                vertices, faces, colors = self.create_circle_mesh(
                    (x, y), radius, height_px, color
                )
            elif shape_type == 'heart':
                vertices, faces, colors = self.create_heart_mesh(
                    params, height_px, color
                )
            elif shape_type == 'fraction':
                vertices, faces, colors = self.create_fraction_mesh(
                    params, height_px, color
                )
            elif shape_type == 'triangle':
                if self.pyramid_mode:
                    vertices, faces, colors = self.create_pyramid_mesh(
                        params, height_px, color
                    )
                else:
                    vertices, faces, colors = self.create_triangle_mesh(
                        params, height_px, color
                    )
            elif shape_type in ['diamond', 'quadrilateral', 'rectangle']:
                if self.diamond_mode:
                    vertices, faces, colors = self.create_diamond_mesh(
                        params, height_px * 1.5, color
                    )
                elif shape_type == 'rectangle' and self.true_3d_mode:
                    vertices, faces, colors = self.create_rectangle_mesh(
                        params, height_px, color
                    )
                else:
                    vertices, faces, colors = self.create_polygon_mesh(
                        params, height_px, color
                    )
            else:  # polygon or any other shape
                vertices_2d = params
                vertices, faces, colors = self.create_polygon_mesh(
                    vertices_2d, height_px, color
                )
            
            if vertices and faces:
                if self.inflation_enabled:
                    vertices, faces, colors = self.inflate_mesh(vertices, faces, colors)
            
            faces = [[idx + face_offset for idx in face] for face in faces]
            face_offset += len(vertices)
            
            all_vertices.extend(vertices)
            all_faces.extend(faces)
            all_colors.extend(colors)
        
        if not all_vertices:
            return None
            
        vertices = np.array(all_vertices, dtype=np.float32)
        faces = np.array(all_faces, dtype=np.uint32)
        colors = np.array(all_colors, dtype=np.float32)
        
        if colors.shape[1] == 3:
            colors = np.column_stack([colors, np.ones(len(colors))])
        
        vertices[:, 0] -= np.mean(vertices[:, 0])
        vertices[:, 1] -= np.mean(vertices[:, 1])
        vertices[:, 2] -= np.mean(vertices[:, 2])
        
        max_dim = np.max(np.ptp(vertices, axis=0))
        if max_dim > 0:
            vertices /= max_dim
            
        mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            face_colors=colors
        )

        if self.smoothing_factor > 0:
            mesh = self.smooth_mesh(mesh, self.smoothing_factor)
        
        return mesh
    
    def advanced_edge_smoothing(self, mesh, smoothing_iterations=5, preserve_features=True):
        """
        Advanced mesh edge smoothing algorithm
        
        Args:
            mesh (trimesh.Trimesh): Input mesh to be smoothed
            smoothing_iterations (int): Number of smoothing iterations
            preserve_features (bool): Whether to preserve sharp features
        
        Returns:
            trimesh.Trimesh: Smoothed mesh
        """
        try:
            # Create a copy of the mesh to avoid modifying the original
            smoothed_mesh = mesh.copy()
            vertices = smoothed_mesh.vertices.copy()
            
            # Compute vertex adjacency and normals
            adjacency = smoothed_mesh.vertex_neighbors
            vertex_normals = smoothed_mesh.vertex_normals
            
            # Compute edge information
            edge_lengths = np.linalg.norm(
                vertices[smoothed_mesh.edges[:, 0]] - vertices[smoothed_mesh.edges[:, 1]], 
                axis=1
            )
            median_edge_length = np.median(edge_lengths)
            
            for iteration in range(smoothing_iterations):
                # Compute vertex offsets
                offsets = np.zeros_like(vertices)
                
                for i in range(len(vertices)):
                    # Get neighboring vertices
                    neighbors = adjacency[i]
                    
                    if not neighbors:
                        continue
                    
                    # Compute neighbor vertices and their positions
                    neighbor_verts = vertices[neighbors]
                    
                    # Compute centroid of neighboring vertices
                    centroid = np.mean(neighbor_verts, axis=0)
                    
                    # Compute displacement vector
                    displacement = centroid - vertices[i]
                    
                    # Feature preservation
                    if preserve_features:
                        # Compute angle between vertex normal and displacement
                        vertex_normal = vertex_normals[i]
                        angle = np.arccos(np.clip(
                            np.dot(displacement, vertex_normal) / 
                            (np.linalg.norm(displacement) * np.linalg.norm(vertex_normal) + 1e-8), 
                            -1.0, 1.0
                        ))
                        
                        # Adjust smoothing based on angle
                        smoothing_factor = (1 - np.abs(angle) / np.pi) ** 2
                        
                        # Adaptive smoothing strength
                        edge_variation = np.std(np.linalg.norm(neighbor_verts - vertices[i], axis=1))
                        adaptive_strength = min(1.0, edge_variation / median_edge_length)
                        
                        # Combine smoothing factors
                        smoothing_strength = smoothing_factor * adaptive_strength * 0.5
                    else:
                        smoothing_strength = 0.5
                    
                    # Apply smoothing offset
                    offsets[i] = displacement * smoothing_strength
                
                # Update vertex positions
                vertices += offsets
            
            # Update mesh vertices
            smoothed_mesh.vertices = vertices
            
            return smoothed_mesh
        
        except Exception as e:
            print(f"Advanced edge smoothing failed: {e}")
            import traceback
            traceback.print_exc()
            return mesh

    def smooth_mesh(self, mesh, factor):
        """
        Enhanced mesh smoothing method
        
        Args:
            mesh (trimesh.Trimesh): Input mesh to be smoothed
            factor (float): Smoothing intensity factor (0-1)
        
        Returns:
            trimesh.Trimesh: Smoothed mesh
        """
        try:
            if factor < 0.05:
                return mesh
            
            # Determine number of iterations based on smoothing factor
            iterations = max(1, min(10, int(factor * 10)))
            
            # Check mesh validity
            if not mesh.is_watertight:
                print("Warning: Mesh is not watertight, using simplified smoothing")
                return self.simple_smooth_mesh(mesh, factor)
            
            if np.any(np.isnan(mesh.vertices)) or np.any(np.isinf(mesh.vertices)):
                print("Warning: Mesh contains invalid coordinates, skipping smoothing")
                return mesh
            
            # Identify and protect center vertices
            vertex_face_count = np.zeros(len(mesh.vertices), dtype=np.int32)
            for face in mesh.faces:
                for vertex in face:
                    vertex_face_count[vertex] += 1
            
            center_vertices = np.where(vertex_face_count > np.mean(vertex_face_count) * 1.5)[0]
            
            # Store original center vertex heights
            original_centers = {idx: mesh.vertices[idx, 2] for idx in center_vertices}
            
            # Apply advanced edge smoothing
            smoothed_mesh = self.advanced_edge_smoothing(
                mesh, 
                smoothing_iterations=iterations, 
                preserve_features=True
            )
            
            # Restore center vertex heights
            for idx, z_val in original_centers.items():
                smoothed_mesh.vertices[idx, 2] = z_val
            
            return smoothed_mesh
        
        except Exception as e:
            print(f"Enhanced smoothing failed: {e}")
            return mesh
    
    def simple_smooth_mesh_with_center_protection(self, mesh, factor, center_vertices=None):
        try:
            smoothed = mesh.copy()
            vertices = smoothed.vertices.copy()
            faces = smoothed.faces
            
            if center_vertices is None:
                center_vertices = {}
                vertex_face_count = np.zeros(len(vertices), dtype=np.int32)
                for face in faces:
                    for vertex in face:
                        vertex_face_count[vertex] += 1
                        
                potential_centers = np.where(vertex_face_count > np.mean(vertex_face_count) * 1.5)[0]
                
                for idx in potential_centers:
                    center_vertices[idx] = vertices[idx, 2]
            
            neighbors = [[] for _ in range(len(vertices))]
            for face in faces:
                for i in range(3):
                    neighbors[face[i]].extend([face[(i+1)%3], face[(i+2)%3]])
            
            for i in range(len(neighbors)):
                neighbors[i] = list(set(neighbors[i]))
            
            strength = min(0.9, factor)
            new_vertices = vertices.copy()
            
            for i in range(len(vertices)):
                if not neighbors[i]:
                    continue
                    
                if i in center_vertices:
                    avg_pos = np.mean([vertices[n] for n in neighbors[i]], axis=0)
                    new_vertices[i][0] = vertices[i][0] * (1 - strength) + avg_pos[0] * strength
                    new_vertices[i][1] = vertices[i][1] * (1 - strength) + avg_pos[1] * strength
                    new_vertices[i][2] = center_vertices[i]
                else:
                    avg_pos = np.mean([vertices[n] for n in neighbors[i]], axis=0)
                    new_vertices[i] = vertices[i] * (1 - strength) + avg_pos * strength
            
            smoothed.vertices = new_vertices
            return smoothed
        except Exception as e:
            print(f"Simple smoothing failed: {e}")
            return mesh

    def simple_smooth_mesh(self, mesh, factor):
        try:
            smoothed = mesh.copy()
            vertices = smoothed.vertices.copy()
            faces = smoothed.faces
            
            neighbors = [[] for _ in range(len(vertices))]
            for face in faces:
                for i in range(3):
                    neighbors[face[i]].extend([face[(i+1)%3], face[(i+2)%3]])
            
            for i in range(len(neighbors)):
                neighbors[i] = list(set(neighbors[i]))
            
            strength = min(0.9, factor)
            new_vertices = vertices.copy()
            
            for i in range(len(vertices)):
                if not neighbors[i]:
                    continue
                    
                avg_pos = np.mean([vertices[n] for n in neighbors[i]], axis=0)
                new_vertices[i] = vertices[i] * (1 - strength) + avg_pos * strength
            
            smoothed.vertices = new_vertices
            return smoothed
        except Exception as e:
            print(f"Simple smoothing failed: {e}")
            return mesh




class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("2D to 3D Shape Converter")
        self.converter = Shape3DConverter()
        self.current_mesh = None
        self.shapes = None
        self.pyramid_mode = False
        self.processed_image = None
        self.original_image = None
        self.smoothing_factor = 0.0
        self.inflation_distribution = 0.0
        self.dimensions = {'width': 100, 'height': 100, 'depth': 10}  # Default in mm
        self.current_unit = 'mm'
        
        app_icon = QIcon()
        icon_path = "logo/OneUp logo-02.png"
        if os.path.exists(icon_path):
            app_icon.addFile(icon_path)
            self.setWindowIcon(app_icon)
        else:
            print(f"Warning: Icon file not found at {icon_path}")
            
        self.setWindowTitle("2D to 3D Shape Converter")
        self.converter = Shape3DConverter()
        self.current_mesh = None
        self.shapes = None
        self.processed_image = None
        self.original_image = None
        self.smoothing_factor = 0.0
        self.inflation_distribution = 0.0
        self.dimensions = {'width': 100, 'height': 100, 'depth': 10}  # Default in mm
        self.current_unit = 'mm'
        
        
        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)
        
        # Left panel
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Back button
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
        
        # Image controls
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
        
        # Dimension controls
        dimension_group = QGroupBox("Dimensions")
        dimension_layout = QVBoxLayout()
        
        # Unit selection
        unit_widget = QWidget()
        unit_layout = QHBoxLayout(unit_widget)
        unit_layout.setContentsMargins(0, 0, 0, 0)
        unit_layout.addWidget(QLabel("Units:"))
        self.unit_combo = QComboBox()
        self.unit_combo.addItems(['mm', 'cm', 'm', 'in', 'ft'])
        self.unit_combo.setCurrentText('mm')
        self.unit_combo.currentTextChanged.connect(self.change_units)
        unit_layout.addWidget(self.unit_combo)
        dimension_layout.addWidget(unit_widget)
        
        # Width control
        width_widget = QWidget()
        width_layout = QHBoxLayout(width_widget)
        width_layout.setContentsMargins(0, 0, 0, 0)
        width_layout.addWidget(QLabel("Width:"))
        self.width_spin = QDoubleSpinBox()
        self.width_spin.setRange(0.1, 1000)
        self.width_spin.setValue(100)
        self.width_spin.setSuffix(" mm")
        self.width_spin.valueChanged.connect(self.update_dimensions)
        width_layout.addWidget(self.width_spin)
        dimension_layout.addWidget(width_widget)
        
        # Height control
        height_widget = QWidget()
        height_layout = QHBoxLayout(height_widget)
        height_layout.setContentsMargins(0, 0, 0, 0)
        height_layout.addWidget(QLabel("Height:"))
        self.height_spin = QDoubleSpinBox()
        self.height_spin.setRange(0.1, 1000)
        self.height_spin.setValue(100)
        self.height_spin.setSuffix(" mm")
        self.height_spin.valueChanged.connect(self.update_dimensions)
        height_layout.addWidget(self.height_spin)
        dimension_layout.addWidget(height_widget)
        
        # Depth control
        depth_widget = QWidget()
        depth_layout = QHBoxLayout(depth_widget)
        depth_layout.setContentsMargins(0, 0, 0, 0)
        depth_layout.addWidget(QLabel("Extrusion:"))
        self.depth_spin = QDoubleSpinBox()
        self.depth_spin.setRange(0.1, 1000)
        self.depth_spin.setValue(10)
        self.depth_spin.setSuffix(" mm")
        self.depth_spin.valueChanged.connect(self.update_dimensions)
        depth_layout.addWidget(self.depth_spin)
        dimension_layout.addWidget(depth_widget)
        
        dimension_group.setLayout(dimension_layout)
        
        # 3D Options
        options_group = QGroupBox("3D Options")
        options_layout = QVBoxLayout()
        
        # Mode selection
        mode_widget = QWidget()
        mode_layout = QHBoxLayout(mode_widget)
        mode_layout.setContentsMargins(0, 0, 0, 0)
        
        self.true_3d_checkbox = QCheckBox("Sphere/Cube")
        self.true_3d_checkbox.setToolTip("Convert to volumetric 3D models")
        self.true_3d_checkbox.stateChanged.connect(self.toggle_true_3d_mode)
        
        self.pyramid_checkbox = QCheckBox("Pyramid Mode")
        self.pyramid_checkbox.setToolTip("Convert triangles to 3D pyramids")
        self.pyramid_checkbox.stateChanged.connect(self.toggle_pyramid_mode)
        options_layout.addWidget(self.pyramid_checkbox)

        # In the options_group section of init_ui
        self.star_checkbox = QCheckBox("Star Mode")
        self.star_checkbox.setToolTip("Convert stars to proper 3D star shapes")
        self.star_checkbox.stateChanged.connect(self.toggle_star_mode)
        mode_layout.addWidget(self.star_checkbox)

        #self.diamond_checkbox = QCheckBox("Diamond Mode")
        #self.diamond_checkbox.setToolTip("Convert quadrilaterals to 3D diamonds")
        #self.diamond_checkbox.stateChanged.connect(self.toggle_diamond_mode)
        #mode_layout.addWidget(self.diamond_checkbox)



        
        #self.inflation_checkbox = QCheckBox("Inflate Shapes")
        #self.inflation_checkbox.setToolTip("Create rounded, inflated versions")
        #self.inflation_checkbox.stateChanged.connect(self.toggle_inflation_mode)
        
        mode_layout.addWidget(self.true_3d_checkbox)
        #mode_layout.addWidget(self.diamond_checkbox)  # Add diamond checkbox to layout
        options_layout.addWidget(mode_widget)
        
        # Inflation controls
        inflation_control = QWidget()
        inflation_layout = QVBoxLayout(inflation_control)
        inflation_layout.setContentsMargins(0, 0, 0, 0)
        
        self.inflation_slider = QSlider(Qt.Orientation.Horizontal)
        self.inflation_slider.setRange(0, 100)
        self.inflation_slider.setValue(50)
        self.inflation_slider.setEnabled(False)
        self.inflation_label = QLabel("Inflation: 50%")
        
        #inflation_layout.addWidget(QLabel("Inflation Amount:"))
        #inflation_layout.addWidget(self.inflation_slider)
        #inflation_layout.addWidget(self.inflation_label)
        #options_layout.addWidget(inflation_control)
        
        # Distribution control
        distribution_control = QWidget()
        distribution_layout = QVBoxLayout(distribution_control)
        distribution_layout.setContentsMargins(0, 0, 0, 0)
        
        #self.distribution_slider = QSlider(Qt.Orientation.Horizontal)
        #self.distribution_slider.setRange(0, 100)
        #self.distribution_slider.setValue(50)
        #self.distribution_slider.setEnabled(True)
        #self.distribution_label = QLabel("Inflation Distribution: 50%")
        
        #distribution_layout.addWidget(QLabel("Inflation Distribution:"))
        #distribution_layout.addWidget(self.distribution_slider)
        #distribution_layout.addWidget(self.distribution_label)
        #options_layout.addWidget(distribution_control)
        
        # Smoothing
        #self.smoothing_checkbox = QCheckBox("Smooth Shapes")
        #self.smoothing_checkbox.stateChanged.connect(self.toggle_smoothing)
        #options_layout.addWidget(self.smoothing_checkbox)
        
        options_group.setLayout(options_layout)
        
        left_layout.addWidget(self.image_label)
        left_layout.addWidget(control_group)
        left_layout.addWidget(dimension_group)
        left_layout.addWidget(options_group)
        left_layout.addStretch()
        
        # 3D Viewer
        self.viewer = gl.GLViewWidget()
        self.viewer.setCameraPosition(distance=1)
        grid = gl.GLGridItem()
        grid.setSize(1, 1)
        grid.setSpacing(0.1, 0.1)
        self.viewer.addItem(grid)
        
        main_layout.addWidget(left_panel, 1)
        main_layout.addWidget(self.viewer, 2)
        self.setCentralWidget(main_widget)

        # Connect signals
        #self.inflation_slider.valueChanged.connect(self.update_inflation)
        #self.distribution_slider.valueChanged.connect(self.update_distribution)
        #self.inflation_checkbox.stateChanged.connect(
        #    lambda state: self.distribution_slider.setEnabled(state == Qt.CheckState.Checked)
        #)

    def toggle_star_mode(self, state):
        """Toggle star mode for star shapes"""
        # In PyQt6, CheckState is an enum, need to compare with its value
        is_enabled = state == Qt.CheckState.Checked.value
        print(f"Star mode toggled: {is_enabled}")
        
        self.converter.set_star_mode(is_enabled)
        
        # If we have shapes and one of them is a star, update the 3D model
        if self.shapes:
            has_star = any(shape[0] == 'star' for shape in self.shapes)
            if has_star:
                print("Star shape found, updating 3D model...")
                self.update_3d_model()

    def toggle_diamond_mode(self, state):
        """Toggle diamond mode for quadrilaterals"""
        is_enabled = state == Qt.CheckState.Checked.value
        self.converter.set_diamond_mode(is_enabled)
        
        # If we have shapes and one of them is a quadrilateral/diamond,
        # update the 3D model immediately
        if self.shapes:
            has_quadrilateral = any(shape[0] in ['quadrilateral', 'diamond', 'rectangle'] 
                                for shape in self.shapes)
            if has_quadrilateral:
                self.update_3d_model()
        
    def toggle_pyramid_mode(self, state):
        """Toggle pyramid mode for triangles"""
        is_enabled = state == Qt.CheckState.Checked.value
        self.converter.set_pyramid_mode(is_enabled)
        
        # Update the model if shapes are already detected
        if self.shapes:
            self.update_3d_model()

    def change_units(self, unit):
        """Handle unit system change"""
        self.current_unit = unit
        suffix = f" {unit}"
        
        # Update spin boxes
        self.width_spin.setSuffix(suffix)
        self.height_spin.setSuffix(suffix)
        self.depth_spin.setSuffix(suffix)
        
        # Convert values if needed
        if unit == 'mm':
            factor = 1.0
        elif unit == 'cm':
            factor = 0.1
        elif unit == 'm':
            factor = 0.001
        elif unit == 'in':
            factor = 1/25.4
        elif unit == 'ft':
            factor = 1/304.8
            
        # Convert current values
        self.width_spin.setValue(self.width_spin.value() * factor)
        self.height_spin.setValue(self.height_spin.value() * factor)
        self.depth_spin.setValue(self.depth_spin.value() * factor)
        
        # Update model if we have one
        if self.shapes:
            self.update_3d_model()

    def update_dimensions(self):
        """Update dimension values when spin boxes change"""
        self.dimensions = {
            'width': self.width_spin.value(),
            'height': self.height_spin.value(),
            'depth': self.depth_spin.value()
        }
        
        if self.shapes:
            self.update_3d_model()

    def get_scale_factor(self):
        """Calculate scale factor based on image dimensions and target size"""
        if self.original_image is None:
            return 1.0
            
        img_height, img_width = self.original_image.shape[:2]
        width_scale = self.convert_to_mm(self.dimensions['width']) / img_width
        height_scale = self.convert_to_mm(self.dimensions['height']) / img_height
        
        return min(width_scale, height_scale)

    def convert_to_mm(self, value):
        """Convert value from current unit to mm"""
        if self.current_unit == 'mm':
            return value
        elif self.current_unit == 'cm':
            return value * 10
        elif self.current_unit == 'm':
            return value * 1000
        elif self.current_unit == 'in':
            return value * 25.4
        elif self.current_unit == 'ft':
            return value * 304.8
        return value

    def update_3d_model(self):
        """Update the 3D model using current settings"""
        try:
            if not self.shapes:
                return
                
            scale_factor = self.get_scale_factor()
            depth_mm = self.convert_to_mm(self.dimensions['depth'])
            normalized_depth = depth_mm / (100 * scale_factor)
            
            self.current_mesh = self.converter.create_3d_mesh(
                self.processed_image if self.processed_image is not None else self.original_image, 
                self.shapes, 
                normalized_depth
            )
            
            if self.current_mesh is None:
                QMessageBox.warning(self, "Error", "Failed to create 3D mesh")
                return
                
            self.current_mesh.apply_scale(scale_factor)
            self.display_mesh(self.current_mesh)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"3D conversion failed: {str(e)}")
            import traceback
            print(f"Error in update_3d_model: {traceback.format_exc()}")

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

    def toggle_smoothing(self, state):
        """Toggle smoothing on/off"""
        is_enabled = state == Qt.CheckState.Checked.value
        
        # Update smoothing factor based on checkbox
        self.smoothing_factor = 0.5 if is_enabled else 0.0
        self.converter.set_smoothing_factor(self.smoothing_factor)
        
        # Update the model if shapes are already detected
        if self.shapes:
            self.update_3d_model()

    def update_distribution(self, value):
        """Update inflation distribution when slider changes"""
        # Map 0-100 to 0.1-2.0 for more noticeable effect
        distribution_factor = 0.1 + (value / 100.0) * 1.9
        self.distribution_label.setText(f"Inflation Distribution: {value}%")
        self.converter.set_inflation_distribution(distribution_factor)
        
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
        
        # Update label based on mode
        if is_3d_mode:
            self.height_label.setText(f"Volume: {self.height_slider.value()/100:.2f}")
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
        else:
            self.height_label.setText(f"Extrusion Height: {value/100:.2f}")

        self.converter.set_extrusion_strength(value/100.0)
        
        # Update 3D model if we have shapes
        if self.shapes:
            self.update_3d_model()

    def display_mesh(self, mesh):
        """Display mesh with enhanced visualization for better 3D appearance"""
        self.viewer.clear()
        
        try:
            vertices = mesh.vertices
            faces = mesh.faces
            
            # Get colors from mesh
            if hasattr(mesh.visual, 'face_colors'):
                colors = mesh.visual.face_colors
                if colors.max() > 1.0:
                    colors = colors.astype(np.float32) / 255.0
            else:
                colors = np.ones((len(faces), 4)) * [0.5, 0.5, 0.5, 1.0]
            
            # Create the mesh item with improved rendering settings
            mesh_item = gl.GLMeshItem(
                vertexes=vertices,
                faces=faces,
                faceColors=colors,
                smooth=True,  # Enable smooth shading
                shader='shaded',  # Use shaded shader for better 3D appearance
                glOptions='opaque',
                drawEdges=False,  # Turn on edges for better visibility
                edgeColor=(0.3, 0.3, 0.3, 0.8)  # Dark gray edges
            )
                
            self.viewer.addItem(mesh_item)
            
            # Set background color to black for better contrast
            self.viewer.setBackgroundColor('black')
            
            # Find the minimum z value to place grid at
            z_min = np.min(vertices[:, 2])
            
            # Add a more detailed grid for better spatial reference
            grid = gl.GLGridItem()
            x_min, y_min, _ = np.min(vertices, axis=0)
            x_max, y_max, _ = np.max(vertices, axis=0)
            grid_size = max(abs(x_max - x_min), abs(y_max - y_min)) * 3.0
            grid.setSize(grid_size, grid_size)
            grid.setSpacing(grid_size/20, grid_size/20)
            grid.translate(0, 0, z_min - 0.02)
            grid.setColor((0.5, 0.5, 0.5, 0.5))  # Gray grid
            self.viewer.addItem(grid)
            
            # Add coordinate axes for reference
            axis_length = grid_size * 0.3
            origin = [0, 0, z_min]
            
            x_axis = gl.GLLinePlotItem(
                pos=np.array([origin, [axis_length, 0, z_min]]), 
                color=(1, 0, 0, 1), width=2  # Red for X-axis
            )
            y_axis = gl.GLLinePlotItem(
                pos=np.array([origin, [0, axis_length, z_min]]), 
                color=(0, 1, 0, 1), width=2  # Green for Y-axis
            )
            z_axis = gl.GLLinePlotItem(
                pos=np.array([origin, [0, 0, z_min + axis_length]]), 
                color=(0, 0, 1, 1), width=2  # Blue for Z-axis
            )
            
            self.viewer.addItem(x_axis)
            self.viewer.addItem(y_axis)
            self.viewer.addItem(z_axis)
            
            # Set camera position for optimal viewing
            mesh_size = np.max(np.ptp(vertices, axis=0))
            camera_distance = max(1.5, mesh_size * 2.5)
            
            # Set more top-down view to better see the star shape
            self.viewer.setCameraPosition(
                distance=camera_distance, 
                elevation=40,  # Higher elevation to see shape better
                azimuth=45   # Rotated view angle
            )
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to display mesh: {str(e)}")
            print(f"Error in display_mesh: {e}")
            import traceback
            traceback.print_exc()
                
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


    def go_back_to_landing_page(self):
        """Close this window and launch the landing page without showing terminal"""
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
                
                # Start the landing page script without showing console
                if sys.platform == 'win32':
                    # Windows-specific solution to hide console
                    startupinfo = subprocess.STARTUPINFO()
                    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                    startupinfo.wShowWindow = subprocess.SW_HIDE
                    subprocess.Popen([sys.executable, landing_page_script],
                                startupinfo=startupinfo,
                                creationflags=subprocess.DETACHED_PROCESS)
                else:
                    # For macOS and Linux
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

def main():
    app = QApplication(sys.argv)
    
    app_icon = QIcon()
    icon_path = "logo/OneUp logo-02.png"
    if os.path.exists(icon_path):
        app_icon.addFile(icon_path)
        app.setWindowIcon(app_icon)
    
    window = MainWindow()
    window.showMaximized()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()