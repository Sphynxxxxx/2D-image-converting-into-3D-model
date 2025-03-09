import sys
import cv2
import numpy as np
import trimesh
import pyqtgraph
from PIL import Image
from rembg import remove
import io
import math
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                           QPushButton, QLabel, QFileDialog, QMessageBox, QGroupBox, QComboBox,
                           QSlider, QProgressBar, QCheckBox)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt, QThread, pyqtSignal
import pyqtgraph.opengl as gl

class RembgBackgroundRemover:
    """Background removal using the rembg library"""
    
    def __init__(self):
        pass
    
    def remove_background(self, image):
        """Remove background using rembg

        Args:
            image: OpenCV image in BGR format
            
        Returns:
            tuple: (image with transparent background in BGRA format, binary mask)
        """
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

class AdvancedDepthEstimator:
    """Advanced depth estimation for better 3D model generation"""
    
    def __init__(self):
        # Default parameters
        self.depth_methods = ['distance_transform', 'normal_based', 'photometric', 'hybrid']
        self.current_method = 'hybrid'
        self.detail_level = 5  # 1-10 scale
        self.smoothness = 5    # 1-10 scale
        self.extrusion_depth = 0.5  # 0.1-1.0 scale
        
    def set_parameters(self, method=None, detail_level=None, smoothness=None, extrusion_depth=None):
        """Set depth estimation parameters"""
        if method is not None and method in self.depth_methods:
            self.current_method = method
        if detail_level is not None:
            self.detail_level = max(1, min(10, detail_level))
        if smoothness is not None:
            self.smoothness = max(1, min(10, smoothness))
        if extrusion_depth is not None:
            self.extrusion_depth = max(0.1, min(1.0, extrusion_depth))
            
    def estimate_depth(self, image, mask):
        """Estimate depth using the selected method
        
        Args:
            image: BGR image
            mask: Binary mask of the object
            
        Returns:
            Depth map as a float32 image with values in range [0.0, 1.0]
        """
        if self.current_method == 'distance_transform':
            return self._distance_transform_depth(image, mask)
        elif self.current_method == 'normal_based':
            return self._normal_based_depth(image, mask)
        elif self.current_method == 'photometric':
            return self._photometric_depth(image, mask)
        else:  # hybrid (default)
            return self._hybrid_depth(image, mask)
            
    def _distance_transform_depth(self, image, mask):
        """Estimate depth using distance transform"""
        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Get edges with Canny
        edges = cv2.Canny(gray, 50, 150)
        
        # Combine with mask edges
        mask_edges = cv2.Canny(mask, 50, 150)
        combined_edges = cv2.bitwise_or(edges, mask_edges)
        
        # Dilate edges based on detail level
        kernel_size = max(1, 11 - self.detail_level)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        dilated_edges = cv2.dilate(combined_edges, kernel, iterations=1)
        
        # Invert for distance transform
        edge_mask = cv2.bitwise_not(dilated_edges)
        
        # Apply distance transform
        edge_mask = edge_mask & mask  # Only consider edges inside the mask
        dist = cv2.distanceTransform(edge_mask, cv2.DIST_L2, 5)
        
        # Normalize and apply extrusion depth
        dist = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)
        
        # Apply smoothing based on smoothness parameter
        blur_size = max(3, self.smoothness * 2 + 1)
        if blur_size % 2 == 0:
            blur_size += 1  # Ensure odd kernel size
        smooth_dist = cv2.GaussianBlur(dist, (blur_size, blur_size), 0)
        
        # Scale by extrusion depth
        depth_map = smooth_dist * self.extrusion_depth
        
        return depth_map
        
    def _normal_based_depth(self, image, mask):
        """Estimate depth based on surface normals"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate gradients
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        
        # Normalize gradients
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        max_mag = np.max(magnitude) if np.max(magnitude) > 0 else 1.0
        grad_x = grad_x / max_mag
        grad_y = grad_y / max_mag
        
        # Convert gradients to surface normals
        normals_z = 1.0 / np.sqrt(1.0 + grad_x**2 + grad_y**2)
        normals_x = -grad_x * normals_z
        normals_y = -grad_y * normals_z
        
        # Convert normals to depth
        depth_map = (normals_z - np.min(normals_z)) / (np.max(normals_z) - np.min(normals_z) + 1e-6)
        
        # Apply mask
        depth_map = depth_map * (mask > 0)
        
        # Apply detail level
        if self.detail_level < 5:
            # Less detail: more smoothing
            blur_size = 11 - self.detail_level * 2
            if blur_size % 2 == 0:
                blur_size += 1
            depth_map = cv2.GaussianBlur(depth_map, (blur_size, blur_size), 0)
        elif self.detail_level > 5:
            # More detail: enhance edges
            detail_factor = (self.detail_level - 5) / 5.0
            edges = cv2.Canny(gray, 50, 150)
            edge_depth = 1.0 - cv2.distanceTransform(cv2.bitwise_not(edges), cv2.DIST_L2, 5) / 255.0
            edge_depth = cv2.normalize(edge_depth, None, 0, 1.0, cv2.NORM_MINMAX)
            depth_map = depth_map * (1.0 - detail_factor) + edge_depth * detail_factor
        
        # Scale by extrusion depth
        depth_map = depth_map * self.extrusion_depth
        
        return depth_map
        
    def _photometric_depth(self, image, mask):
        """Estimate depth using photometric cues (brightness/shadows)"""
        # Convert to grayscale and normalize
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        
        # Enhance contrast with CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply((gray * 255).astype(np.uint8)).astype(np.float32) / 255.0
        
        # Apply detail level - adjust local contrast
        if self.detail_level > 5:
            # Increase local contrast for more detail
            detail_factor = (self.detail_level - 5) / 5.0
            enhanced = enhanced * (1.0 + detail_factor)
            enhanced = np.clip(enhanced, 0.0, 1.0)
        elif self.detail_level < 5:
            # Reduce local contrast for less detail
            detail_factor = (5 - self.detail_level) / 5.0
            mean_value = np.mean(enhanced)
            enhanced = enhanced * (1.0 - detail_factor) + mean_value * detail_factor
        
        # Invert: darker areas are higher (typical for shading)
        depth_map = 1.0 - enhanced
        
        # Apply mask
        depth_map = depth_map * (mask > 0)
        
        # Apply smoothing based on smoothness parameter
        blur_size = max(3, self.smoothness * 2 + 1)
        if blur_size % 2 == 0:
            blur_size += 1  # Ensure odd kernel size
        depth_map = cv2.GaussianBlur(depth_map, (blur_size, blur_size), 0)
        
        # Scale by extrusion depth
        depth_map = depth_map * self.extrusion_depth
        
        return depth_map
        
    def _hybrid_depth(self, image, mask):
        """Combine multiple depth estimation methods for better results"""
        # Get depth maps from different methods
        dist_depth = self._distance_transform_depth(image, mask)
        normal_depth = self._normal_based_depth(image, mask)
        photo_depth = self._photometric_depth(image, mask)
        
        # Define weights based on detail level
        if self.detail_level < 4:
            # Favor distance transform for smoother results
            dist_weight = 0.6
            normal_weight = 0.2
            photo_weight = 0.2
        elif self.detail_level < 7:
            # Balanced approach
            dist_weight = 0.4
            normal_weight = 0.3
            photo_weight = 0.3
        else:
            # Favor normal and photometric for detailed results
            dist_weight = 0.2
            normal_weight = 0.4
            photo_weight = 0.4
            
        # Combine depth maps
        combined_depth = (
            dist_depth * dist_weight + 
            normal_depth * normal_weight + 
            photo_depth * photo_weight
        )
        
        # Ensure proper range
        combined_depth = np.clip(combined_depth, 0.0, 1.0)
        
        # Apply final smoothing if needed
        if self.smoothness > 7:
            blur_size = self.smoothness
            if blur_size % 2 == 0:
                blur_size += 1
            combined_depth = cv2.GaussianBlur(combined_depth, (blur_size, blur_size), 0)
            
        return combined_depth

class AdvancedMeshGenerator(QThread):
    """Advanced mesh generator for better 3D models"""
    finished = pyqtSignal(np.ndarray, np.ndarray, np.ndarray)
    mesh_ready = pyqtSignal(trimesh.Trimesh)
    progress_update = pyqtSignal(int, str)  # Progress percentage, status message
    
    def __init__(self, image, mask=None, depth_estimator=None):
        super().__init__()
        self.image = image
        self.mask = mask
        
        # Use provided depth estimator or create a default one
        self.depth_estimator = depth_estimator if depth_estimator is not None else AdvancedDepthEstimator()
        
        # Mesh generation parameters
        self.mesh_quality = 5  # 1-10 scale
        self.use_subdivision = True
        self.subdivide_iterations = 1
        self.triangulation_method = 'delaunay'  # 'delaunay' or 'grid'
        self.smooth_normals = True
        self.add_base = True
        self.base_thickness = 0.1  # As fraction of max height
        
    def set_parameters(self, mesh_quality=None, use_subdivision=None, 
                       subdivide_iterations=None, triangulation_method=None,
                       smooth_normals=None, add_base=None, base_thickness=None):
        """Set mesh generation parameters"""
        if mesh_quality is not None:
            self.mesh_quality = max(1, min(10, mesh_quality))
        if use_subdivision is not None:
            self.use_subdivision = use_subdivision
        if subdivide_iterations is not None:
            self.subdivide_iterations = max(0, min(3, subdivide_iterations))
        if triangulation_method is not None and triangulation_method in ['delaunay', 'grid']:
            self.triangulation_method = triangulation_method
        if smooth_normals is not None:
            self.smooth_normals = smooth_normals
        if add_base is not None:
            self.add_base = add_base
        if base_thickness is not None:
            self.base_thickness = max(0.05, min(0.5, base_thickness))
            
    def detect_object(self, image):
        """Detect object in image if no mask is provided"""
        if self.mask is not None:
            return self.mask
            
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Use adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create mask
        mask = np.zeros_like(gray)
        
        if contours:
            # Filter small contours
            min_area = (gray.shape[0] * gray.shape[1]) * 0.01  # 1% of image area
            valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
            
            if valid_contours:
                # Find largest contour
                largest_contour = max(valid_contours, key=cv2.contourArea)
                
                # Get convex hull
                hull = cv2.convexHull(largest_contour)
                
                # Approximate contour for smoother shape
                epsilon = 0.02 * cv2.arcLength(hull, True)
                approx_contour = cv2.approxPolyDP(hull, epsilon, True)
                
                # Fill mask
                cv2.drawContours(mask, [approx_contour], -1, 255, -1)
                
                # Clean up with morphological operations
                kernel = np.ones((5, 5), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask
            
    def generate_mesh(self, image, object_mask, depth_map):
        """Generate 3D mesh from image, mask, and depth map"""
        self.progress_update.emit(10, "Creating base mesh...")
        
        # Set mesh resolution based on quality
        if self.mesh_quality <= 3:
            # Low quality, fewer vertices
            downsample_factor = 4
        elif self.mesh_quality <= 7:
            # Medium quality
            downsample_factor = 2
        else:
            # High quality, more vertices
            downsample_factor = 1
            
        # Downsample image, mask, and depth map for performance
        h, w = depth_map.shape
        target_h, target_w = h // downsample_factor, w // downsample_factor
        
        # Resize image, mask, and depth map
        image_small = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)
        mask_small = cv2.resize(object_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        depth_small = cv2.resize(depth_map, (target_w, target_h), interpolation=cv2.INTER_AREA)
        
        # Create vertices and faces
        vertices = []
        colors = []
        
        # Masked area indices
        y_indices, x_indices = np.where(mask_small > 0)
        
        self.progress_update.emit(20, "Creating vertices...")
        
        # Grid method
        if self.triangulation_method == 'grid':
            # Create mesh grid
            h, w = mask_small.shape
            vertex_map = np.full((h, w), -1)  # Map to track vertex indices
            current_vertex = 0
            
            # Generate vertices
            for i in range(h):
                for j in range(w):
                    if mask_small[i, j] > 0:
                        vertices.append([
                            (j - w/2) / max(w, h),  # Normalized x
                            (i - h/2) / max(w, h),  # Normalized y
                            depth_small[i, j]      # z from depth map
                        ])
                        
                        # Store vertex color
                        colors.append(image_small[i, j] / 255.0)
                        
                        # Store vertex index in map
                        vertex_map[i, j] = current_vertex
                        current_vertex += 1
            
            self.progress_update.emit(40, "Creating faces...")
            
            # Generate faces
            faces = []
            for i in range(h - 1):
                for j in range(w - 1):
                    if (mask_small[i, j] > 0 and mask_small[i+1, j] > 0 and 
                        mask_small[i, j+1] > 0 and mask_small[i+1, j+1] > 0):
                        
                        # Get vertex indices
                        v1 = vertex_map[i, j]
                        v2 = vertex_map[i, j+1]
                        v3 = vertex_map[i+1, j]
                        v4 = vertex_map[i+1, j+1]
                        
                        if all(v != -1 for v in [v1, v2, v3, v4]):
                            # Create two triangles
                            faces.append([v1, v2, v3])
                            faces.append([v3, v2, v4])
        
        # Delaunay triangulation method
        else:
            # Get points inside mask
            xy_points = np.column_stack((x_indices, y_indices))
            
            # Generate vertices for all points
            for i, (x, y) in enumerate(xy_points):
                vertices.append([
                    (x - target_w/2) / max(target_w, target_h),  # Normalized x
                    (y - target_h/2) / max(target_w, target_h),  # Normalized y
                    depth_small[y, x]  # z from depth map
                ])
                
                # Store vertex color
                colors.append(image_small[y, x] / 255.0)
            
            self.progress_update.emit(40, "Creating Delaunay triangulation...")
            
            # Create triangulation from points
            rect = (0, 0, target_w, target_h)
            subdiv = cv2.Subdiv2D(rect)
            
            for point in xy_points:
                subdiv.insert((float(point[0]), float(point[1])))
            
            # Get triangles from subdivision
            triangles = subdiv.getTriangleList()
            
            # Build a map from (x,y) to vertex index
            point_to_index = {}
            for i, (x, y) in enumerate(xy_points):
                point_to_index[(x, y)] = i
            
            # Create faces
            faces = []
            for triangle in triangles:
                x1, y1, x2, y2, x3, y3 = map(int, triangle)
                
                # Check if all points are inside the mask
                if (mask_small[y1, x1] > 0 and 
                    mask_small[y2, x2] > 0 and 
                    mask_small[y3, x3] > 0):
                    
                    # Get vertex indices
                    try:
                        v1 = point_to_index.get((x1, y1))
                        v2 = point_to_index.get((x2, y2))
                        v3 = point_to_index.get((x3, y3))
                        
                        if v1 is not None and v2 is not None and v3 is not None:
                            faces.append([v1, v2, v3])
                    except:
                        continue  # Skip problematic triangles
        
        self.progress_update.emit(60, "Creating mesh...")
        
        # Convert to numpy arrays
        vertices = np.array(vertices, dtype=np.float32)
        faces = np.array(faces, dtype=np.uint32)
        colors = np.array(colors, dtype=np.float32)
        
        if len(faces) == 0 or len(vertices) == 0:
            return None, np.array([]), np.array([]), np.array([])
        
        # Create back face (base) if requested
        if self.add_base:
            self.progress_update.emit(70, "Adding base...")
            
            # Calculate base height
            min_z = np.min(vertices[:, 2])
            base_z = min_z - self.base_thickness
            
            # Get unique xy coordinates from border
            border_vertices = []
            border_indices = []
            
            # Method depends on triangulation type
            if self.triangulation_method == 'grid':
                # Find border pixels in the mask
                border_mask = mask_small.copy()
                eroded = cv2.erode(border_mask, np.ones((3, 3), np.uint8), iterations=1)
                border = border_mask - eroded
                
                border_y, border_x = np.where(border > 0)
                
                # Get vertex indices for border points
                for y, x in zip(border_y, border_x):
                    idx = vertex_map[y, x]
                    if idx != -1:
                        border_indices.append(idx)
                        border_vertices.append(vertices[idx])
                
            else:  # Delaunay
                # Find border vertices using alpha shape concept
                from scipy.spatial import Delaunay
                
                # Extract xy coordinates
                xy_coords = vertices[:, :2]
                
                # Compute Delaunay triangulation
                tri = Delaunay(xy_coords)
                
                # Find border edges
                edges = set()
                for simplex in tri.simplices:
                    edges.add((simplex[0], simplex[1]))
                    edges.add((simplex[1], simplex[2]))
                    edges.add((simplex[2], simplex[0]))
                
                # Count occurrences of each edge
                from collections import Counter
                edge_count = Counter()
                for edge in edges:
                    # Sort edge indices to avoid duplicates
                    sorted_edge = tuple(sorted(edge))
                    edge_count[sorted_edge] += 1
                
                # Boundary edges appear only once
                boundary_edges = [edge for edge, count in edge_count.items() if count == 1]
                
                # Get unique boundary vertices
                boundary_vertices = set()
                for edge in boundary_edges:
                    boundary_vertices.add(edge[0])
                    boundary_vertices.add(edge[1])
                
                border_indices = list(boundary_vertices)
                border_vertices = [vertices[idx] for idx in border_indices]
            
            # Sort border vertices to form a loop
            if len(border_vertices) > 2:
                # Start with a border vertex
                ordered_border = [0]
                remaining = set(range(1, len(border_vertices)))
                current = 0
                
                while remaining:
                    # Find nearest neighbor
                    current_point = np.array(border_vertices[ordered_border[-1]])[:2]  # xy only
                    min_dist = float('inf')
                    nearest = None
                    
                    for idx in remaining:
                        point = np.array(border_vertices[idx])[:2]  # xy only
                        dist = np.sum((current_point - point) ** 2)
                        if dist < min_dist:
                            min_dist = dist
                            nearest = idx
                    
                    if nearest is not None:
                        ordered_border.append(nearest)
                        remaining.remove(nearest)
                    else:
                        break  # Shouldn't happen but just in case
                
                # Create base vertices
                num_existing = len(vertices)
                base_vertices = []
                for idx in ordered_border:
                    v = border_vertices[idx].copy()
                    v[2] = base_z  # Set to base height
                    base_vertices.append(v)
                
                # Add base vertices to the main vertices array
                vertices = np.vstack([vertices, base_vertices])
                
                # Add colors for base vertices (use border colors or single color)
                base_color = np.array([0.8, 0.8, 0.8])  # Light gray
                base_colors = np.tile(base_color, (len(base_vertices), 1))
                colors = np.vstack([colors, base_colors])
                
                # Create triangles for base (use simple triangle fan)
                base_faces = []
                for i in range(1, len(ordered_border) - 1):
                    base_faces.append([
                        num_existing + ordered_border[0],
                        num_existing + ordered_border[i],
                        num_existing + ordered_border[i + 1]
                    ])
                
                # Create triangles for sides (connecting top and bottom)
                side_faces = []
                for i in range(len(ordered_border)):
                    next_i = (i + 1) % len(ordered_border)
                    top_idx = border_indices[ordered_border[i]]
                    next_top_idx = border_indices[ordered_border[next_i]]
                    bottom_idx = num_existing + ordered_border[i]
                    next_bottom_idx = num_existing + ordered_border[next_i]
                    
                    # Create two triangles for each quad
                    side_faces.append([top_idx, next_top_idx, bottom_idx])
                    side_faces.append([bottom_idx, next_top_idx, next_bottom_idx])
                
                # Add base and side faces to the main faces array
                if base_faces:
                    faces = np.vstack([faces, base_faces])
                if side_faces:
                    faces = np.vstack([faces, side_faces])
        
        self.progress_update.emit(80, "Creating trimesh object...")
        
        # Create mesh
        mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            vertex_colors=(colors * 255).astype(np.uint8)
        )
        
        # Fixed code for subdivision section in the generate_mesh method:

        # Apply subdivision if enabled
        if self.use_subdivision and self.subdivide_iterations > 0:
            self.progress_update.emit(85, f"Applying subdivision (iterations: {self.subdivide_iterations})...")
            
            try:
                # Use trimesh's subdivision
                for _ in range(self.subdivide_iterations):
                    mesh = mesh.subdivide()
            except Exception as e:
                print(f"Subdivision failed: {str(e)}")

        # Smooth normals
        if self.smooth_normals:
            self.progress_update.emit(90, "Smoothing normals...")
            mesh.smooth_shaded = True  # Just set this flag for trimesh to compute smooth normals

        # Clean up mesh
        self.progress_update.emit(95, "Cleaning up mesh...")
        try:
            mesh.remove_degenerate_faces()
            mesh.remove_duplicate_faces()
            mesh.remove_unreferenced_vertices()
        except Exception as e:
            print(f"Mesh cleanup failed: {str(e)}")

        # Update mesh data after cleanup
        vertices = mesh.vertices.astype(np.float32)
        faces = mesh.faces.astype(np.uint32)
        colors = (mesh.visual.vertex_colors[:len(vertices)] / 255.0).astype(np.float32)

        self.progress_update.emit(100, "Mesh generation complete!")

        return mesh, vertices, faces, colors

    # Fix for the RemoveBackgroundThread class:
    class RemoveBackgroundThread(QThread):
        """Thread for removing background using rembg"""
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

    # Fix for the AdvancedMeshGenerator class (add run method):
    def run(self):
        try:
            self.progress_update.emit(0, "Starting mesh generation...")
            
            # Detect object if mask not provided
            if self.mask is None:
                self.progress_update.emit(5, "Detecting object...")
                object_mask = self.detect_object(self.image)
            else:
                object_mask = self.mask
            
            # Estimate depth
            self.progress_update.emit(8, "Estimating depth map...")
            depth_map = self.depth_estimator.estimate_depth(self.image, object_mask)
            
            # Generate mesh
            mesh, vertices, faces, colors = self.generate_mesh(self.image, object_mask, depth_map)
            
            # Emit results
            if mesh is not None:
                self.mesh_ready.emit(mesh)
                self.finished.emit(vertices, faces, colors)
            else:
                empty_mesh = trimesh.Trimesh()
                self.mesh_ready.emit(empty_mesh)
                self.finished.emit(np.array([]), np.array([]), np.array([]))
                
        except Exception as e:
            print(f"Error in mesh generation: {str(e)}")
            self.progress_update.emit(100, f"Error: {str(e)}")
            empty_mesh = trimesh.Trimesh()
            self.mesh_ready.emit(empty_mesh)
            self.finished.emit(np.array([]), np.array([]), np.array([]))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Blender-like 3D Model Generator")
        self.setGeometry(100, 100, 1200, 800)
        self.setMinimumSize(1000, 700)
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
            QSlider::groove:horizontal {
                border: 1px solid #999999;
                height: 8px;
                background: #424242;
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #5c5c5c;
                border: 1px solid #777;
                width: 16px;
                margin: -2px 0;
                border-radius: 4px;
            }
            QProgressBar {
                border: 1px solid #424242;
                border-radius: 3px;
                background-color: #1e1e1e;
                text-align: center;
                height: 18px;
            }
            QProgressBar::chunk {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                                              stop:0 #3a7ebf, 
                                              stop:1 #4d9eff);
                border-radius: 2px;
            }
            QComboBox {
                border: 1px solid #424242;
                border-radius: 3px;
                padding: 5px;
                background-color: #363636;
            }
            QCheckBox {
                spacing: 5px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
            }
            QCheckBox::indicator:unchecked {
                border: 1px solid #777;
                background-color: #363636;
                border-radius: 2px;
            }
            QCheckBox::indicator:checked {
                border: 1px solid #777;
                background-color: #4d9eff;
                border-radius: 2px;
            }
        """)
        
        # Initialize components
        self.init_components()
        
        self.init_ui()
        self.image_path = None
        self.current_mesh = None

    def init_components(self):
        """Initialize components needed for 3D conversion"""
        # Background removal
        self.bg_remover = RembgBackgroundRemover()
        self.original_image = None
        self.processed_image = None
        self.mask = None
        
        # Depth estimator
        self.depth_estimator = AdvancedDepthEstimator()
        
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
        """Create background removal UI group"""
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

    def add_depth_settings_ui(self):
        """Create depth estimation settings UI group"""
        depth_group = QGroupBox("Depth Settings")
        depth_group.setStyleSheet("""
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
        depth_layout = QVBoxLayout(depth_group)
        
        # Depth method selection
        method_layout = QHBoxLayout()
        method_label = QLabel("Depth Method:")
        self.depth_method_combo = QComboBox()
        self.depth_method_combo.addItems(["Hybrid", "Distance Transform", "Normal Based", "Photometric"])
        self.depth_method_combo.currentTextChanged.connect(self.update_depth_settings)
        method_layout.addWidget(method_label)
        method_layout.addWidget(self.depth_method_combo)
        
        # Detail level slider
        detail_layout = QHBoxLayout()
        detail_label = QLabel("Detail Level:")
        self.detail_slider = QSlider(Qt.Orientation.Horizontal)
        self.detail_slider.setRange(1, 10)
        self.detail_slider.setValue(5)
        self.detail_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.detail_slider.setTickInterval(1)
        self.detail_slider.valueChanged.connect(self.update_depth_settings)
        detail_layout.addWidget(detail_label)
        detail_layout.addWidget(self.detail_slider)
        self.detail_value_label = QLabel("5")
        detail_layout.addWidget(self.detail_value_label)
        
        # Smoothness slider
        smooth_layout = QHBoxLayout()
        smooth_label = QLabel("Smoothness:")
        self.smooth_slider = QSlider(Qt.Orientation.Horizontal)
        self.smooth_slider.setRange(1, 10)
        self.smooth_slider.setValue(5)
        self.smooth_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.smooth_slider.setTickInterval(1)
        self.smooth_slider.valueChanged.connect(self.update_depth_settings)
        smooth_layout.addWidget(smooth_label)
        smooth_layout.addWidget(self.smooth_slider)
        self.smooth_value_label = QLabel("5")
        smooth_layout.addWidget(self.smooth_value_label)
        
        # Extrusion depth slider
        extrusion_layout = QHBoxLayout()
        extrusion_label = QLabel("Extrusion:")
        self.extrusion_slider = QSlider(Qt.Orientation.Horizontal)
        self.extrusion_slider.setRange(1, 10)
        self.extrusion_slider.setValue(5)
        self.extrusion_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.extrusion_slider.setTickInterval(1)
        self.extrusion_slider.valueChanged.connect(self.update_depth_settings)
        extrusion_layout.addWidget(extrusion_label)
        extrusion_layout.addWidget(self.extrusion_slider)
        self.extrusion_value_label = QLabel("0.5")
        extrusion_layout.addWidget(self.extrusion_value_label)
        
        # Add all layouts to main layout
        depth_layout.addLayout(method_layout)
        depth_layout.addLayout(detail_layout)
        depth_layout.addLayout(smooth_layout)
        depth_layout.addLayout(extrusion_layout)
        
        return depth_group

    def add_mesh_settings_ui(self):
        """Create mesh generation settings UI group"""
        mesh_group = QGroupBox("Mesh Settings")
        mesh_group.setStyleSheet("""
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
        mesh_layout = QVBoxLayout(mesh_group)
        
        # Mesh quality slider
        quality_layout = QHBoxLayout()
        quality_label = QLabel("Mesh Quality:")
        self.quality_slider = QSlider(Qt.Orientation.Horizontal)
        self.quality_slider.setRange(1, 10)
        self.quality_slider.setValue(5)
        self.quality_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.quality_slider.setTickInterval(1)
        self.quality_slider.valueChanged.connect(self.update_mesh_settings)
        quality_layout.addWidget(quality_label)
        quality_layout.addWidget(self.quality_slider)
        self.quality_value_label = QLabel("Medium")
        quality_layout.addWidget(self.quality_value_label)
        
        # Triangulation method
        triangulation_layout = QHBoxLayout()
        triangulation_label = QLabel("Triangulation:")
        self.triangulation_combo = QComboBox()
        self.triangulation_combo.addItems(["Delaunay", "Grid"])
        self.triangulation_combo.currentTextChanged.connect(self.update_mesh_settings)
        triangulation_layout.addWidget(triangulation_label)
        triangulation_layout.addWidget(self.triangulation_combo)
        
        # Subdivision checkbox
        subdivision_layout = QHBoxLayout()
        self.subdivision_check = QCheckBox("Use Subdivision")
        self.subdivision_check.setChecked(True)
        self.subdivision_check.stateChanged.connect(self.update_mesh_settings)
        subdivision_layout.addWidget(self.subdivision_check)
        
        # Subdivision iterations slider (only enabled if subdivision is checked)
        self.subdivision_slider = QSlider(Qt.Orientation.Horizontal)
        self.subdivision_slider.setRange(1, 3)
        self.subdivision_slider.setValue(1)
        self.subdivision_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.subdivision_slider.setTickInterval(1)
        self.subdivision_slider.valueChanged.connect(self.update_mesh_settings)
        self.subdivision_value_label = QLabel("1")
        subdivision_layout.addWidget(self.subdivision_slider)
        subdivision_layout.addWidget(self.subdivision_value_label)
        
        # Smooth normals checkbox
        normals_layout = QHBoxLayout()
        self.normals_check = QCheckBox("Smooth Normals")
        self.normals_check.setChecked(True)
        self.normals_check.stateChanged.connect(self.update_mesh_settings)
        normals_layout.addWidget(self.normals_check)
        
        # Add base checkbox and thickness
        base_layout = QHBoxLayout()
        self.base_check = QCheckBox("Add Base")
        self.base_check.setChecked(True)
        self.base_check.stateChanged.connect(self.update_mesh_settings)
        base_layout.addWidget(self.base_check)
        
        self.base_slider = QSlider(Qt.Orientation.Horizontal)
        self.base_slider.setRange(1, 10)
        self.base_slider.setValue(2)
        self.base_slider.valueChanged.connect(self.update_mesh_settings)
        self.base_value_label = QLabel("0.1")
        base_layout.addWidget(self.base_slider)
        base_layout.addWidget(self.base_value_label)
        
        # Add all layouts to main layout
        mesh_layout.addLayout(quality_layout)
        mesh_layout.addLayout(triangulation_layout)
        mesh_layout.addLayout(subdivision_layout)
        mesh_layout.addLayout(normals_layout)
        mesh_layout.addLayout(base_layout)
        
        return mesh_group

    def init_ui(self):
        """Initialize the main UI"""
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
        
        # Depth settings group
        depth_settings_group = self.add_depth_settings_ui()
        
        # Mesh settings group
        mesh_settings_group = self.add_mesh_settings_ui()
        
        # Progress bar
        progress_group = QGroupBox("Progress")
        progress_group.setStyleSheet("""
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
        progress_layout = QVBoxLayout(progress_group)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        
        self.status_label = QLabel("Ready")
        
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.status_label)
        
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
        self.select_button.clicked.connect(self.select_image)
        
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
        
        controls_layout.addWidget(self.select_button)
        controls_layout.addWidget(self.convert_button)
        controls_layout.addWidget(self.export_button)
        
        # Create a tab widget for settings to save space
        settings_tabs = QGroupBox("Settings")
        settings_tabs.setStyleSheet("""
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
        settings_layout = QVBoxLayout(settings_tabs)
        
        # Add settings groups to layout
        settings_layout.addWidget(depth_settings_group)
        settings_layout.addWidget(mesh_settings_group)
        
        # Add groups to left panel
        left_layout.addWidget(image_group)
        left_layout.addWidget(bg_removal_group)
        left_layout.addWidget(settings_tabs)
        left_layout.addWidget(progress_group)
        left_layout.addWidget(controls_group)
        
        # Create scrollable area for left panel
        scroll = QWidget()
        scroll_layout = QVBoxLayout(scroll)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.addWidget(left_panel)
        scroll_layout.addStretch()
        
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
        self.viewer.setCameraPosition(distance=3.0, elevation=30, azimuth=45)
        self.viewer.setBackgroundColor('#1e1e1e')  # Dark background
        
        # Add grid for better perspective
        grid = gl.GLGridItem()
        grid.setSize(x=10, y=10, z=1)
        grid.setSpacing(x=0.5, y=0.5, z=0.5)
        grid.setColor((0.3, 0.3, 0.3, 1.0))  # Dark gray grid
        self.viewer.addItem(grid)
        
        # Add lights for better visibility
        self.viewer.setCameraPosition(distance=3.0, elevation=30, azimuth=45)
        self.viewer.setCameraPosition(distance=3.0, elevation=30, azimuth=45)

        viewer_layout.addWidget(self.viewer)
        
        # Add camera controls information
        camera_info = QLabel("Camera Controls: Left-click & drag to rotate, Right-click & drag to pan, Scroll to zoom")
        camera_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        camera_info.setStyleSheet("color: #888888; font-size: 10px;")
        viewer_layout.addWidget(camera_info)
        
        # Add panels to content layout
        content_layout.addWidget(scroll, 1)
        content_layout.addWidget(viewer_group, 2)
        
        # Add content layout to main layout
        main_layout.addLayout(content_layout)
        
        self.setCentralWidget(main_widget)
        
        # Initialize settings
        self.update_depth_settings()
        self.update_mesh_settings()

    def update_depth_settings(self):
        """Update depth estimation settings based on UI controls"""
        # Get method
        method = self.depth_method_combo.currentText().lower().replace(" ", "_")
        
        # Get detail level
        detail = self.detail_slider.value()
        self.detail_value_label.setText(str(detail))
        
        # Get smoothness
        smoothness = self.smooth_slider.value()
        self.smooth_value_label.setText(str(smoothness))
        
        # Get extrusion depth
        extrusion = self.extrusion_slider.value() / 10.0
        self.extrusion_value_label.setText(f"{extrusion:.1f}")
        
        # Update depth estimator
        self.depth_estimator.set_parameters(
            method=method,
            detail_level=detail,
            smoothness=smoothness,
            extrusion_depth=extrusion
        )

    def update_mesh_settings(self):
        """Update mesh generation settings based on UI controls"""
        # Get mesh quality
        quality = self.quality_slider.value()
        quality_text = "Low"
        if quality >= 4 and quality <= 7:
            quality_text = "Medium"
        elif quality > 7:
            quality_text = "High"
        self.quality_value_label.setText(quality_text)
        
        # Get triangulation method
        triangulation = self.triangulation_combo.currentText().lower()
        
        # Get subdivision settings
        use_subdivision = self.subdivision_check.isChecked()
        self.subdivision_slider.setEnabled(use_subdivision)
        
        subdivide_iterations = self.subdivision_slider.value()
        self.subdivision_value_label.setText(str(subdivide_iterations))
        
        # Get normal smoothing
        smooth_normals = self.normals_check.isChecked()
        
        # Get base settings
        add_base = self.base_check.isChecked()
        self.base_slider.setEnabled(add_base)
        
        base_thickness = self.base_slider.value() / 20.0  # Scale 1-10 to 0.05-0.5
        self.base_value_label.setText(f"{base_thickness:.2f}")
        
        # These settings will be used when creating the mesh generator

    def select_image(self):
        """Select an image file"""
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
            self.mask = None
            
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
            
            # Reset progress
            self.progress_bar.setValue(0)
            self.status_label.setText("Ready")

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
        
        # Update status
        self.status_label.setText("Background removed successfully")

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
        
        # Update status
        self.status_label.setText(f"Error: {error_message}")

    def remove_background(self):
        """Remove background using rembg"""
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
        
        # Update status
        self.status_label.setText("Removing background...")
        
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
            
            # Reset processed image and mask
            self.processed_image = None
            self.mask = None
            
            # Disable reset button
            self.reset_image_button.setEnabled(False)
            
            # Update status
            self.status_label.setText("Image reset to original")

    def update_progress(self, value, message):
        """Update progress bar and status message"""
        self.progress_bar.setValue(value)
        self.status_label.setText(message)

    def convert_to_3d(self):
        """Convert image to 3D model"""
        if not self.image_path:
            return
            
        try:
            # Use processed image if available, otherwise use original
            if hasattr(self, 'processed_image') and self.processed_image is not None:
                # If BGRA, convert to BGR by dropping alpha channel
                if self.processed_image.shape[2] == 4:
                    image = cv2.cvtColor(self.processed_image, cv2.COLOR_BGRA2BGR)
                else:
                    image = self.processed_image.copy()
                mask = self.mask
            else:
                image = cv2.imread(self.image_path)
                self.original_image = image.copy()
                mask = None
            
            if image is None:
                raise ValueError("Failed to load image")
                
            # Update depth estimation settings before conversion
            self.update_depth_settings()
            
            # Create mesh generator with current settings
            mesh_generator = AdvancedMeshGenerator(image, mask, self.depth_estimator)
            
            # Set mesh generation parameters based on UI settings
            mesh_generator.set_parameters(
                mesh_quality=self.quality_slider.value(),
                use_subdivision=self.subdivision_check.isChecked(),
                subdivide_iterations=self.subdivision_slider.value(),
                triangulation_method=self.triangulation_combo.currentText().lower(),
                smooth_normals=self.normals_check.isChecked(),
                add_base=self.base_check.isChecked(),
                base_thickness=self.base_slider.value() / 20.0
            )
            
            # Connect signals
            mesh_generator.finished.connect(self.display_mesh)
            mesh_generator.mesh_ready.connect(self.store_mesh)
            mesh_generator.progress_update.connect(self.update_progress)
            
            # Disable buttons during processing
            self.convert_button.setEnabled(False)
            self.export_button.setEnabled(False)
            
            # Start conversion
            self.status_label.setText("Starting 3D conversion...")
            self.progress_bar.setValue(0)
            mesh_generator.start()
            
        except Exception as e:
            print(f"Error in convert_to_3d: {str(e)}")
            QMessageBox.critical(
                self,
                "Conversion Error",
                f"Failed to process the image: {str(e)}\n\nPlease try a different image or settings."
            )
            self.status_label.setText(f"Error: {str(e)}")
            
            # Re-enable buttons
            self.convert_button.setEnabled(True)

    def display_mesh(self, vertices, faces, colors):
        """Display the 3D mesh in the viewer"""
        self.viewer.clear()
        
        try:
            if len(faces) > 0 and len(vertices) > 0:
                # Basic data validation
                if len(vertices) < 3 or len(faces) < 1:
                    raise ValueError("Not enough vertices or faces")

                # Ensure proper data types and shapes
                vertices = np.array(vertices, dtype=np.float32)
                faces = np.array(faces, dtype=np.uint32)
                
                # Center and normalize the model for better display
                center = np.mean(vertices, axis=0)
                vertices = vertices - center
                
                max_dim = np.max(np.abs(vertices))
                if max_dim > 0:
                    vertices = vertices / max_dim
                
                # Create simple monochrome colors if color data is invalid
                if len(colors) != len(faces) or colors.shape[1] != 3:
                    colors = np.ones((len(faces), 4), dtype=np.float32) * [0.7, 0.7, 0.7, 1.0]
                else:
                    # Convert RGB to RGBA
                    alpha = np.ones((len(colors), 1), dtype=np.float32)
                    colors = np.hstack([colors, alpha]).astype(np.float32)

                # Create mesh with improved parameters for Blender-like rendering
                mesh = gl.GLMeshItem(
                    vertexes=vertices,
                    faces=faces,
                    faceColors=colors,
                    smooth=self.normals_check.isChecked(),  # Use smooth shading if enabled
                    drawEdges=False,  # No edge lines for more realistic look
                    shader='shaded',  # Use shaded mode for better 3D appearance
                    glOptions='opaque'  # No transparency
                )
                
                # Add reference grid for scale
                grid = gl.GLGridItem()
                grid.setSize(x=2, y=2, z=0)
                grid.setSpacing(x=0.1, y=0.1, z=0.1)
                grid.translate(0, 0, -1)  # Move grid below the model
                
                # Add items to viewer
                self.viewer.addItem(grid)
                self.viewer.addItem(mesh)
                
                # Set camera position for better view
                self.viewer.setCameraPosition(distance=4.0, elevation=30, azimuth=45)
                
                # Set viewer options
                self.viewer.opts['fov'] = 40  # Narrower FOV for more realistic perspective
                
                # Update view
                self.viewer.update()
                
                # Show success message
                self.status_label.setText("3D model created successfully")
                
                # Enable export button
                self.export_button.setEnabled(True)
            else:
                QMessageBox.warning(
                    self,
                    "Generation Failed",
                    "Failed to create a valid 3D model. Try adjusting settings or using a different image."
                )
                self.status_label.setText("Failed to create valid 3D model")
                
            # Re-enable convert button
            self.convert_button.setEnabled(True)
            
        except Exception as e:
            print(f"Error displaying mesh: {str(e)}")
            QMessageBox.warning(
                self,
                "Display Error",
                f"Failed to display the 3D model: {str(e)}\n\nPlease try with different settings."
            )
            self.status_label.setText(f"Error: {str(e)}")
            self.convert_button.setEnabled(True)
    
    def export_mesh(self):
        """Export the current mesh to a file"""
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
                
                # Update status
                self.status_label.setText(f"Exporting to {file_name}...")
                
                # Export options
                export_options = {}
                
                # Different options based on file type
                if file_name.lower().endswith('.obj'):
                    # For OBJ, include vertex colors and normals
                    export_options['include_normals'] = True
                    export_options['include_color'] = True
                elif file_name.lower().endswith('.ply'):
                    # For PLY, include vertex colors
                    export_options['encoding'] = 'binary'
                    export_options['vertex_normal'] = True
                    export_options['vertex_color'] = True
                
                # Export the mesh
                self.current_mesh.export(file_name, **export_options)
                
                # Show success message
                QMessageBox.information(
                    self,
                    "Export Success",
                    f"Model exported successfully to:\n{file_name}"
                )
                
                # Update status
                self.status_label.setText(f"Exported to {file_name}")
                
        except Exception as e:
            print(f"Error exporting mesh: {str(e)}")
            QMessageBox.critical(
                self,
                "Export Error",
                f"Failed to export the model: {str(e)}\n\nPlease try again with a different format or location."
            )
            
            # Update status
            self.status_label.setText(f"Export error: {str(e)}")

    def store_mesh(self, mesh):
        """Store the generated mesh for later export"""
        self.current_mesh = mesh
        
        # Enable or disable export button based on mesh validity
        if mesh is not None and len(mesh.faces) > 0:
            self.export_button.setEnabled(True)
        else:
            self.export_button.setEnabled(False)
            print("Warning: Invalid or empty mesh received")


def main():
    app = QApplication(sys.argv)
    
    # Set application style for better appearance
    app.setStyle("Fusion")
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()    