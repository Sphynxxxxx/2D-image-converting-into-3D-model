import sys
import cv2
import numpy as np
import trimesh
from PIL import Image
from rembg import remove
import io
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                           QPushButton, QLabel, QFileDialog, QMessageBox, QGroupBox, QComboBox,
                           QProgressBar, QSlider)
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

class ObjectDetector:
    """Advanced object detection for 3D model generation"""
    
    def __init__(self):
        self.detection_params = {
            'blur_size': 5,
            'canny_low': 30,
            'canny_high': 100,
            'threshold_method': 'adaptive',  # 'otsu', 'adaptive', or 'binary'
            'adaptive_block_size': 11,
            'adaptive_c': 2,
            'binary_threshold': 127,
            'min_area_percent': 1.0,
            'convexity_threshold': 0.8,
            'contour_approximation': 0.02,
            'morph_iterations': 2
        }
    
    def update_params(self, params):
        """Update detection parameters"""
        self.detection_params.update(params)
    
    def detect_object(self, image, mask=None):
        """Detect the main object in the image using advanced methods
        
        Args:
            image: OpenCV image in BGR format
            mask: Optional binary mask from background removal
            
        Returns:
            tuple: (object mask, visualization, detection score, object class)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply blur to reduce noise
        blur_size = self.detection_params['blur_size']
        blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
        
        # Create visualization image for showing detection steps
        visualization = image.copy()
        
        # Use provided mask if available
        if mask is not None:
            # Use mask but refine edges using contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            refined_mask = np.zeros_like(gray)
            
            if contours:
                # Get largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                cv2.drawContours(refined_mask, [largest_contour], -1, 255, -1)
                cv2.drawContours(visualization, [largest_contour], -1, (0, 255, 0), 2)
                
                # Get convex hull for more stable shape
                hull = cv2.convexHull(largest_contour)
                cv2.drawContours(visualization, [hull], -1, (0, 0, 255), 2)
                
                # Automatic object class detection based on shape analysis
                object_class = self.detect_object_class(largest_contour)
                detection_score = 0.9  # High confidence with bg removal
            else:
                refined_mask = mask.copy()
                object_class = "Unknown"
                detection_score = 0.7
            
            return refined_mask, visualization, detection_score, object_class
        
        # If no mask provided, detect object from scratch
        # Try multiple methods and combine them
        masks = []
        
        # Method 1: Edge-based detection
        canny_low = self.detection_params['canny_low']
        canny_high = self.detection_params['canny_high']
        edges = cv2.Canny(blurred, canny_low, canny_high)
        
        # Close the edges to form a continuous boundary
        kernel = np.ones((5, 5), np.uint8)
        closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, 
                                       iterations=self.detection_params['morph_iterations'])
        
        # Fill the closed edges to get a mask
        edge_mask = np.zeros_like(gray)
        contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Filter contours by size
            min_area = (gray.shape[0] * gray.shape[1]) * (self.detection_params['min_area_percent'] / 100)
            valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
            
            if valid_contours:
                largest_contour = max(valid_contours, key=cv2.contourArea)
                cv2.drawContours(edge_mask, [largest_contour], -1, 255, -1)
                masks.append(edge_mask)
        
        # Method 2: Threshold-based detection
        threshold_method = self.detection_params['threshold_method']
        threshold_mask = np.zeros_like(gray)
        
        if threshold_method == 'otsu':
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        elif threshold_method == 'adaptive':
            block_size = self.detection_params['adaptive_block_size']
            c_value = self.detection_params['adaptive_c']
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY_INV, block_size, c_value)
        else:  # binary
            binary_threshold = self.detection_params['binary_threshold']
            _, thresh = cv2.threshold(blurred, binary_threshold, 255, cv2.THRESH_BINARY_INV)
        
        # Clean up the threshold mask
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, 
                                iterations=self.detection_params['morph_iterations'])
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, 
                                iterations=self.detection_params['morph_iterations'])
        
        # Find contours in threshold image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Filter contours by size
            min_area = (gray.shape[0] * gray.shape[1]) * (self.detection_params['min_area_percent'] / 100)
            valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
            
            if valid_contours:
                largest_contour = max(valid_contours, key=cv2.contourArea)
                cv2.drawContours(threshold_mask, [largest_contour], -1, 255, -1)
                masks.append(threshold_mask)
        
        # Combine masks if multiple methods were successful
        if masks:
            # Start with the first mask
            combined_mask = masks[0].copy()
            
            # Add additional masks with weights
            for i in range(1, len(masks)):
                combined_mask = cv2.addWeighted(combined_mask, 0.5, masks[i], 0.5, 0)
            
            # Threshold to get binary mask
            _, final_mask = cv2.threshold(combined_mask, 127, 255, cv2.THRESH_BINARY)
            
            # Get contours from combined mask for visualization
            contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Get convex hull
                hull = cv2.convexHull(largest_contour)
                
                # Approximate the contour for smoother shape
                epsilon = self.detection_params['contour_approximation'] * cv2.arcLength(hull, True)
                approx_contour = cv2.approxPolyDP(hull, epsilon, True)
                
                # Draw contours on visualization
                cv2.drawContours(visualization, [largest_contour], -1, (0, 255, 0), 2)
                cv2.drawContours(visualization, [hull], -1, (0, 0, 255), 2)
                cv2.drawContours(visualization, [approx_contour], -1, (255, 0, 0), 2)
                
                # Create final mask from approximated contour
                final_mask = np.zeros_like(gray)
                cv2.drawContours(final_mask, [approx_contour], -1, 255, -1)
                
                # Smoothen mask edges
                final_mask = cv2.GaussianBlur(final_mask, (5, 5), 0)
                _, final_mask = cv2.threshold(final_mask, 127, 255, cv2.THRESH_BINARY)
                
                # Calculate detection score based on contour properties
                area = cv2.contourArea(largest_contour)
                hull_area = cv2.contourArea(hull)
                convexity = area / hull_area if hull_area > 0 else 0
                
                # Score based on convexity and other factors
                convexity_score = min(convexity / self.detection_params['convexity_threshold'], 1.0)
                detection_score = convexity_score * 0.8
                
                # Automatic object class detection
                object_class = self.detect_object_class(largest_contour)
                
                return final_mask, visualization, detection_score, object_class
        
        # Fallback: return a simple mask based on center of the image
        fallback_mask = np.zeros_like(gray)
        center_x, center_y = gray.shape[1] // 2, gray.shape[0] // 2
        radius = min(gray.shape[0], gray.shape[1]) // 4
        cv2.circle(fallback_mask, (center_x, center_y), radius, 255, -1)
        
        return fallback_mask, visualization, 0.2, "Unknown"  # Low confidence score
    
    def detect_object_class(self, contour):
        """Detect the object class based on shape analysis
        
        Args:
            contour: OpenCV contour
            
        Returns:
            str: Detected object class
        """
        # Get basic shape properties
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # Calculate shape descriptors
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
        # Get bounding rectangle and its properties
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h if h > 0 else 0
        
        # Get rotated bounding rectangle
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        rect_area = rect[1][0] * rect[1][1]
        extent = float(area) / rect_area if rect_area > 0 else 0
        
        # Simple shape classification
        if circularity > 0.8:
            return "Circular"
        elif circularity > 0.6:
            return "Rounded"
        elif aspect_ratio > 1.5 or aspect_ratio < 0.67:
            return "Elongated"
        elif extent > 0.8:
            return "Rectangular"
        else:
            return "Complex"

class EnhancedMeshGenerator(QThread):
    finished = pyqtSignal(np.ndarray, np.ndarray, np.ndarray)
    mesh_ready = pyqtSignal(trimesh.Trimesh)
    accuracy_ready = pyqtSignal(float, dict, str)  # Added object class to signal
    detection_visualization = pyqtSignal(np.ndarray)  # Signal for detection visualization

    def __init__(self, image, object_detector, bg_mask=None):
        super().__init__()
        self.image = image
        self.object_detector = object_detector
        self.bg_mask = bg_mask

    def estimate_depth(self, image, object_mask, object_class):
        """Estimate depth map using enhanced techniques based on object class"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Adjust depth estimation based on detected object class
        if object_class == "Circular":
            # For circular objects, use radial gradient for depth
            height, width = gray.shape
            center_y, center_x = np.mean(np.where(object_mask > 0), axis=1)
            y, x = np.ogrid[:height, :width]
            
            # Create a distance map from center
            dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            max_dist = np.max(dist_from_center * (object_mask > 0) / 255)
            
            # Create radial depth map (higher in center, lower at edges)
            depth_map = 1.0 - (dist_from_center / max_dist if max_dist > 0 else 0)
            depth_map = depth_map * (object_mask > 0) / 255
            
            # Apply smoothing
            depth_map = cv2.GaussianBlur(depth_map, (5, 5), 0)
            
        elif object_class == "Rectangular" or object_class == "Elongated":
            # For rectangular objects, use distance transform with edge enhancement
            edges = cv2.Canny(gray, 50, 150)
            edges = cv2.dilate(edges, None, iterations=1)
            
            # Distance transform from edges
            dist_transform = cv2.distanceTransform(cv2.bitwise_not(edges), cv2.DIST_L2, 5)
            dist_transform = cv2.normalize(dist_transform, None, 0, 1.0, cv2.NORM_MINMAX)
            
            # Apply to object mask
            depth_map = dist_transform * (object_mask > 0) / 255
            
        else:
            # For complex or unknown shapes, use a combination of techniques
            # Edge detection with different thresholds for better detail
            edges1 = cv2.Canny(gray, 30, 100)
            edges2 = cv2.Canny(gray, 50, 150)
            edges = cv2.addWeighted(edges1, 0.5, edges2, 0.5, 0)
            
            # Dilate edges to preserve shape boundaries
            dilated_edges = cv2.dilate(edges, None, iterations=2)
            
            # Apply distance transform to get depth from shape boundaries
            dist_transform = cv2.distanceTransform(object_mask, cv2.DIST_L2, 5)
            dist_transform = cv2.normalize(dist_transform, None, 0, 1.0, cv2.NORM_MINMAX)
            
            # Combine distance transform with edge information
            depth_map = dist_transform * 0.7 + cv2.GaussianBlur(dilated_edges.astype(float) / 255, (5, 5), 0) * 0.3
            
            # Apply adaptive histogram equalization for better detail
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            depth_map = clahe.apply((depth_map * 255).astype(np.uint8)).astype(float) / 255
            
            # Final smoothing while preserving edges
            depth_map = cv2.bilateralFilter(depth_map.astype(np.float32), 9, 75, 75)
        
        return depth_map

    def generate_enhanced_mesh(self, image):
        """Generate 3D mesh focusing on the main object"""
        # Detect object and create mask
        object_mask, visualization, detection_score, object_class = self.object_detector.detect_object(image, self.bg_mask)
        
        # Emit detection visualization
        self.detection_visualization.emit(visualization)
        
        # Calculate accuracy metrics
        accuracy_metrics = {
            'detection_score': float(detection_score * 100),
            'object_coverage': float(np.sum(object_mask > 0) / (object_mask.shape[0] * object_mask.shape[1]) * 100),
            'edge_quality': float(self.calculate_edge_quality(object_mask) * 100),
            'shape_complexity': float(self.calculate_shape_complexity(object_mask) * 100)
        }
        
        # Calculate overall accuracy score (weighted average)
        weights = {
            'detection_score': 0.4,
            'object_coverage': 0.2,
            'edge_quality': 0.2,
            'shape_complexity': 0.2
        }
        overall_accuracy = sum(accuracy_metrics[key] * weights[key] for key in weights)
        
        # Emit accuracy metrics
        self.accuracy_ready.emit(overall_accuracy, accuracy_metrics, object_class)
        
        # Get enhanced depth map using the object mask and class
        depth_map = self.estimate_depth(image, object_mask, object_class)
        
        # Apply mask to depth map
        depth_map = depth_map * (object_mask > 0).astype(float)
        
        # Create vertices only for the masked region
        height, width = depth_map.shape
        y, x = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        
        # Initialize vertices array
        vertices = []
        faces = []
        colors = []
        vertex_map = np.full((height, width), -1)  # Map to track vertex indices
        current_vertex = 0
        
        # Generate vertices only for masked region
        for i in range(height):
            for j in range(width):
                if object_mask[i, j] > 0:
                    # Use different depth scaling based on object class
                    depth_scale = 0.4  # Default
                    if object_class == "Circular":
                        depth_scale = 0.5
                    elif object_class == "Rectangular":
                        depth_scale = 0.3
                    
                    vertices.append([
                        (j - width/2) / max(width, height),
                        (i - height/2) / max(width, height),
                        depth_map[i, j] * depth_scale
                    ])
                    vertex_map[i, j] = current_vertex
                    current_vertex += 1
        
        # Generate faces only for valid vertices
        for i in range(height-1):
            for j in range(width-1):
                if (object_mask[i,j] > 0 and object_mask[i+1,j] > 0 and 
                    object_mask[i,j+1] > 0 and object_mask[i+1,j+1] > 0):
                    # Get vertex indices
                    v1 = vertex_map[i,j]
                    v2 = vertex_map[i,j+1]
                    v3 = vertex_map[i+1,j]
                    v4 = vertex_map[i+1,j+1]
                    
                    if all(v != -1 for v in [v1, v2, v3, v4]):
                        # Create two triangles
                        faces.extend([[v1, v2, v3], [v3, v2, v4]])
                        
                        # Calculate color based on image and depth
                        base_color = image[i,j] / 255.0
                        depth_factor = depth_map[i,j]
                        color = base_color * (0.7 + 0.3 * depth_factor)
                        colors.extend([color, color])
        
        # Convert to numpy arrays with proper types
        vertices = np.array(vertices, dtype=np.float32)
        faces = np.array(faces, dtype=np.uint32)
        colors = np.array(colors, dtype=np.float32)
        
        if len(faces) > 0:
            # Create mesh with proper data types
            mesh = trimesh.Trimesh(
                vertices=vertices,
                faces=faces,
                vertex_colors=colors.repeat(3, axis=0)
            )
            
            # Mesh cleanup and optimization
            mesh.remove_degenerate_faces()
            mesh.remove_duplicate_faces()
            mesh.remove_unreferenced_vertices()
            
            # Update mesh data after cleanup
            vertices = mesh.vertices.astype(np.float32)
            faces = mesh.faces.astype(np.uint32)
            colors = (mesh.visual.vertex_colors[:len(faces)] / 255.0).astype(np.float32)
            
            return mesh, vertices, faces, colors, overall_accuracy, accuracy_metrics, object_class
        else:
            return None, vertices, np.array([], dtype=np.uint32), np.array([], dtype=np.float32), 0, accuracy_metrics, "Unknown"
    
    def calculate_edge_quality(self, mask):
        """Calculate the edge quality of the mask"""
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0
        
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Calculate perimeter and area
        perimeter = cv2.arcLength(largest_contour, True)
        area = cv2.contourArea(largest_contour)
        
        # Ideal circle has the minimum perimeter for a given area
        ideal_perimeter = 2 * np.sqrt(np.pi * area)
        
        # Edge quality is the ratio of ideal perimeter to actual perimeter
        # Higher values (closer to 1) indicate smoother edges
        edge_quality = ideal_perimeter / perimeter if perimeter > 0 else 0
        
        return min(edge_quality, 1.0)  # Cap at 1.0
    
    def calculate_shape_complexity(self, mask):
        """Calculate the shape complexity based on contour approximation"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0
        
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Calculate area and convex hull area
        area = cv2.contourArea(largest_contour)
        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)
        
        # Calculate solidity (ratio of contour area to convex hull area)
        solidity = float(area) / hull_area if hull_area > 0 else 0
        
        # Higher solidity (closer to 1) indicates simpler shapes
        return solidity

    def run(self):
        try:
            mesh, vertices, faces, colors, accuracy, accuracy_metrics, object_class = self.generate_enhanced_mesh(self.image)
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

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Model Generator")
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
        """)
        
        # Initialize background removal
        self.init_bg_removal()
        
        # Initialize object detector
        self.object_detector = ObjectDetector()
        
        self.init_ui()
        self.image_path = None
        self.current_mesh = None
        self.detected_object_class = "Unknown"

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
        bg_removal_group = QGroupBox("Background Removal (rembg)")
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

    def add_detection_controls_ui(self):
        """Add UI for object detection controls"""
        detection_group = QGroupBox("Object Detection Settings")
        detection_group.setStyleSheet("""
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
            QSlider {
                height: 20px;
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
            QComboBox {
                border: 1px solid #424242;
                border-radius: 3px;
                padding: 5px;
                background-color: #363636;
            }
        """)
        
        detection_layout = QVBoxLayout(detection_group)
        
        # Threshold method selection
        threshold_layout = QHBoxLayout()
        threshold_label = QLabel("Detection Method:")
        self.threshold_combo = QComboBox()
        self.threshold_combo.addItems(["Adaptive", "Otsu", "Binary"])
        self.threshold_combo.setCurrentText("Adaptive")
        self.threshold_combo.currentTextChanged.connect(self.update_detection_settings)
        threshold_layout.addWidget(threshold_label)
        threshold_layout.addWidget(self.threshold_combo)
        
        # Min area percent slider
        area_layout = QHBoxLayout()
        area_label = QLabel("Min Size (%):")
        self.area_slider = QSlider(Qt.Orientation.Horizontal)
        self.area_slider.setRange(1, 20)
        self.area_slider.setValue(5)
        self.area_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.area_slider.setTickInterval(5)
        self.area_slider.valueChanged.connect(self.update_detection_settings)
        area_layout.addWidget(area_label)
        area_layout.addWidget(self.area_slider)
        self.area_value_label = QLabel("5%")
        area_layout.addWidget(self.area_value_label)
        
        # Detail level slider
        detail_layout = QHBoxLayout()
        detail_label = QLabel("Detail Level:")
        self.detail_slider = QSlider(Qt.Orientation.Horizontal)
        self.detail_slider.setRange(1, 10)
        self.detail_slider.setValue(5)
        self.detail_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.detail_slider.setTickInterval(1)
        self.detail_slider.valueChanged.connect(self.update_detection_settings)
        detail_layout.addWidget(detail_label)
        detail_layout.addWidget(self.detail_slider)
        self.detail_value_label = QLabel("Medium")
        detail_layout.addWidget(self.detail_value_label)
        
        # Add to layout
        detection_layout.addLayout(threshold_layout)
        detection_layout.addLayout(area_layout)
        detection_layout.addLayout(detail_layout)
        
        return detection_group

    def add_accuracy_ui(self):
        """Add UI elements for displaying detection accuracy"""
        accuracy_group = QGroupBox("Object Detection Status")
        accuracy_group.setStyleSheet("""
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
        """)
        
        accuracy_layout = QVBoxLayout(accuracy_group)
        
        # Object class display
        class_layout = QHBoxLayout()
        class_label = QLabel("Detected Object:")
        self.object_class_label = QLabel("Unknown")
        self.object_class_label.setStyleSheet("font-weight: bold; color: #4d9eff;")
        class_layout.addWidget(class_label)
        class_layout.addWidget(self.object_class_label)
        
        # Overall accuracy
        overall_layout = QHBoxLayout()
        self.overall_accuracy_label = QLabel("Detection Accuracy:")
        self.overall_accuracy_value = QLabel("0%")
        self.overall_accuracy_progress = QProgressBar()
        self.overall_accuracy_progress.setRange(0, 100)
        self.overall_accuracy_progress.setValue(0)
        overall_layout.addWidget(self.overall_accuracy_label)
        overall_layout.addWidget(self.overall_accuracy_progress, 1)
        overall_layout.addWidget(self.overall_accuracy_value)
        
        # Add individual metrics
        self.metric_labels = {}
        self.metric_values = {}
        self.metric_progress = {}
        
        metrics = [
            ("detection_score", "Recognition Score"),
            ("object_coverage", "Object Coverage"),
            ("edge_quality", "Edge Quality"),
            ("shape_complexity", "Shape Simplicity")
        ]
        
        metric_layouts = {}
        for metric_id, metric_name in metrics:
            metric_layouts[metric_id] = QHBoxLayout()
            self.metric_labels[metric_id] = QLabel(f"{metric_name}:")
            self.metric_values[metric_id] = QLabel("0%")
            self.metric_progress[metric_id] = QProgressBar()
            self.metric_progress[metric_id].setRange(0, 100)
            self.metric_progress[metric_id].setValue(0)
            
            metric_layouts[metric_id].addWidget(self.metric_labels[metric_id])
            metric_layouts[metric_id].addWidget(self.metric_progress[metric_id], 1)
            metric_layouts[metric_id].addWidget(self.metric_values[metric_id])
        
        # Add all layouts to main layout
        accuracy_layout.addLayout(class_layout)
        accuracy_layout.addLayout(overall_layout)
        for metric_id, _ in metrics:
            accuracy_layout.addLayout(metric_layouts[metric_id])
        
        return accuracy_group

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
        
        # Detection visualization preview (new)
        self.detection_label = QLabel("Object Detection Visualization")
        self.detection_label.setFixedSize(400, 300)
        self.detection_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.detection_label.setStyleSheet("""
            QLabel {
                background-color: #1e1e1e;
                border: 2px dashed #404040;
                border-radius: 4px;
                color: #808080;
            }
        """)
        image_layout.addWidget(self.detection_label)
        
        # Background removal group
        bg_removal_group = self.add_background_removal_ui()
        
        # Object detection controls group (new)
        detection_controls_group = self.add_detection_controls_ui()
        
        # Accuracy display group
        accuracy_group = self.add_accuracy_ui()
        
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
        
        self.detect_button = QPushButton("🔍 Detect Object")
        self.detect_button.setMinimumHeight(40)
        self.detect_button.setEnabled(False)
        self.detect_button.setStyleSheet(self.button_style)
        self.detect_button.clicked.connect(self.detect_object)
        
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
        controls_layout.addWidget(self.detect_button)
        controls_layout.addWidget(self.convert_button)
        controls_layout.addWidget(self.export_button)
        
        # Add groups to left panel
        left_layout.addWidget(image_group)
        left_layout.addWidget(bg_removal_group)
        left_layout.addWidget(detection_controls_group)
        left_layout.addWidget(accuracy_group)
        left_layout.addWidget(controls_group)
        left_layout.addStretch()
        
        # Create scrollable container for left panel
        scroll_area = QWidget()
        scroll_layout = QVBoxLayout(scroll_area)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.addWidget(left_panel)
        
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
        
        viewer_layout.addWidget(self.viewer)
        
        # Add panels to content layout
        content_layout.addWidget(scroll_area, 1)
        content_layout.addWidget(viewer_group, 2)
        
        # Add content layout to main layout
        main_layout.addLayout(content_layout)
        
        self.setCentralWidget(main_widget)

    def update_detection_settings(self):
        """Update detection settings based on UI controls"""
        # Update threshold method
        method = self.threshold_combo.currentText().lower()
        self.object_detector.update_params({'threshold_method': method})
        
        # Update min area
        min_area = self.area_slider.value()
        self.area_value_label.setText(f"{min_area}%")
        self.object_detector.update_params({'min_area_percent': min_area})
        
        # Update detail level
        detail = self.detail_slider.value()
        detail_text = "Low"
        if detail >= 3 and detail <= 7:
            detail_text = "Medium"
        elif detail > 7:
            detail_text = "High"
        self.detail_value_label.setText(detail_text)
        
        # Map detail level to contour approximation and blur size
        contour_approx = 0.04 - (detail * 0.003)  # Higher detail = smaller epsilon
        blur_size = max(3, detail)
        if blur_size % 2 == 0:  # Must be odd
            blur_size += 1
            
        self.object_detector.update_params({
            'contour_approximation': contour_approx,
            'blur_size': blur_size,
            'morph_iterations': max(1, detail // 3)
        })

    def update_accuracy_display(self, overall_accuracy, accuracy_metrics, object_class):
        """Update the accuracy display UI with the calculated metrics"""
        # Update object class
        self.object_class_label.setText(object_class)
        self.detected_object_class = object_class
        
        # Update overall accuracy
        self.overall_accuracy_value.setText(f"{overall_accuracy:.1f}%")
        self.overall_accuracy_progress.setValue(int(overall_accuracy))
        
        # Change progress bar color based on accuracy level
        if overall_accuracy < 40:
            self.overall_accuracy_progress.setStyleSheet(
                "QProgressBar::chunk { background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #c92a2a, stop:1 #e03131); }"
            )
        elif overall_accuracy < 70:
            self.overall_accuracy_progress.setStyleSheet(
                "QProgressBar::chunk { background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #e8b308, stop:1 #fcc419); }"
            )
        else:
            self.overall_accuracy_progress.setStyleSheet(
                "QProgressBar::chunk { background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #37b24d, stop:1 #40c057); }"
            )
        
        # Update individual metrics
        for metric_id, value in accuracy_metrics.items():
            if metric_id in self.metric_values:
                self.metric_values[metric_id].setText(f"{value:.1f}%")
                self.metric_progress[metric_id].setValue(int(value))

    def update_detection_visualization(self, visualization):
        """Update the detection visualization preview"""
        if visualization is not None:
            # Convert BGR to RGB for Qt
            rgb_image = cv2.cvtColor(visualization, cv2.COLOR_BGR2RGB)
            
            # Convert to QPixmap
            height, width, channel = rgb_image.shape
            bytes_per_line = 3 * width
            q_img = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            
            # Display the visualization
            self.detection_label.setPixmap(pixmap.scaled(
                self.detection_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            ))

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
            
            # Reset detection visualization
            self.detection_label.setText("Object Detection Visualization")
            self.detection_label.setPixmap(QPixmap())
            
            # Enable buttons
            self.detect_button.setEnabled(True)
            self.remove_bg_button.setEnabled(True)
            self.reset_image_button.setEnabled(False)
            self.convert_button.setEnabled(False)
            
            # Reset accuracy display
            self.object_class_label.setText("Unknown")
            self.update_accuracy_display(0.0, {
                'detection_score': 0.0,
                'object_coverage': 0.0,
                'edge_quality': 0.0,
                'shape_complexity': 0.0
            }, "Unknown")

    def detect_object(self):
        """Detect object in the current image"""
        if not self.image_path:
            return
            
        try:
            # Use processed image if available, otherwise use original
            if hasattr(self, 'processed_image') and self.processed_image is not None:
                if self.processed_image.shape[2] == 4:  # BGRA
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
            
            # Detect object
            object_mask, visualization, detection_score, object_class = self.object_detector.detect_object(image, mask)
            
            # Update visualization
            self.update_detection_visualization(visualization)
            
            # Calculate accuracy metrics
            accuracy_metrics = {
                'detection_score': float(detection_score * 100),
                'object_coverage': float(np.sum(object_mask > 0) / (object_mask.shape[0] * object_mask.shape[1]) * 100),
                'edge_quality': float(self.calculate_edge_quality(object_mask) * 100),
                'shape_complexity': float(self.calculate_shape_complexity(object_mask) * 100)
            }
            
            # Calculate overall accuracy
            weights = {
                'detection_score': 0.4,
                'object_coverage': 0.2,
                'edge_quality': 0.2,
                'shape_complexity': 0.2
            }
            overall_accuracy = sum(accuracy_metrics[key] * weights[key] for key in weights)
            
            # Update accuracy display
            self.update_accuracy_display(overall_accuracy, accuracy_metrics, object_class)
            
            # Enable 3D conversion if detection is good enough
            if overall_accuracy > 30:
                self.convert_button.setEnabled(True)
            else:
                self.convert_button.setEnabled(False)
                QMessageBox.warning(
                    self,
                    "Low Detection Quality",
                    "The object detection quality is too low for 3D conversion.\n"
                    "Try adjusting detection settings or using a clearer image."
                )
            
        except Exception as e:
            print(f"Error in object detection: {str(e)}")
            QMessageBox.critical(
                self,
                "Detection Error",
                "Failed to detect object. Please try a different image or settings."
            )
    
    def calculate_edge_quality(self, mask):
        """Calculate the edge quality of the mask"""
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0
        
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Calculate perimeter and area
        perimeter = cv2.arcLength(largest_contour, True)
        area = cv2.contourArea(largest_contour)
        
        # Ideal circle has the minimum perimeter for a given area
        ideal_perimeter = 2 * np.sqrt(np.pi * area)
        
        # Edge quality is the ratio of ideal perimeter to actual perimeter
        # Higher values (closer to 1) indicate smoother edges
        edge_quality = ideal_perimeter / perimeter if perimeter > 0 else 0
        
        return min(edge_quality, 1.0)  # Cap at 1.0
    
    def calculate_shape_complexity(self, mask):
        """Calculate the shape complexity based on contour approximation"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0
        
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Calculate area and convex hull area
        area = cv2.contourArea(largest_contour)
        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)
        
        # Calculate solidity (ratio of contour area to convex hull area)
        solidity = float(area) / hull_area if hull_area > 0 else 0
        
        # Higher solidity (closer to 1) indicates simpler shapes
        return solidity

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
        
        # Reset detection visualization
        self.detection_label.setText("Object Detection Visualization")
        self.detection_label.setPixmap(QPixmap())
        
        # Enable reset button and detect button
        self.reset_image_button.setEnabled(True)
        self.remove_bg_button.setEnabled(True)  # Re-enable the button
        self.remove_bg_button.setText("🎭 Remove Background")  # Reset text
        self.detect_button.setEnabled(True)

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
            
            # Reset detection visualization
            self.detection_label.setText("Object Detection Visualization")
            self.detection_label.setPixmap(QPixmap())
            
            # Disable reset button
            self.reset_image_button.setEnabled(False)
            self.convert_button.setEnabled(False)

    def convert_to_3d(self):
        """Convert image to 3D with advanced object detection"""
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
                bg_mask = self.mask
            else:
                image = cv2.imread(self.image_path)
                bg_mask = None
            
            if image is None:
                raise ValueError("Failed to load image")
                
            # Reduce size more aggressively to prevent rendering issues
            max_size = 128  # Smaller size for better performance
            height, width = image.shape[:2]
            scale = max_size / max(height, width)
            new_size = (int(width * scale), int(height * scale))
            image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
            
            if bg_mask is not None:
                bg_mask = cv2.resize(bg_mask, new_size, interpolation=cv2.INTER_NEAREST)
            
            # Disable buttons during processing
            self.convert_button.setEnabled(False)
            self.convert_button.setText("Generating 3D...")
            
            # Start mesh generation with object detector
            self.mesh_thread = EnhancedMeshGenerator(image, self.object_detector, bg_mask)
            self.mesh_thread.finished.connect(self.display_mesh)
            self.mesh_thread.mesh_ready.connect(self.store_mesh)
            self.mesh_thread.accuracy_ready.connect(self.update_accuracy_display)
            self.mesh_thread.detection_visualization.connect(self.update_detection_visualization)
            self.mesh_thread.start()
            
        except Exception as e:
            print(f"Error in convert_to_3d: {str(e)}")
            QMessageBox.critical(
                self,
                "Conversion Error",
                "Failed to process the image. Please try a different image."
            )
            # Re-enable button
            self.convert_button.setEnabled(True)
            self.convert_button.setText("🔄 Generate 3D Model")

    def display_mesh(self, vertices, faces, colors):
        """Display the 3D mesh with simplified rendering"""
        self.viewer.clear()
        
        try:
            if len(faces) > 0 and len(vertices) > 0:
                # Basic data validation
                if len(vertices) < 3 or len(faces) < 1:
                    raise ValueError("Not enough vertices or faces")

                # Ensure proper data types and shapes
                vertices = np.array(vertices, dtype=np.float32)
                faces = np.array(faces, dtype=np.uint32)
                
                # Normalize vertices to prevent rendering issues
                scale = 1.0 / max(abs(vertices.min()), abs(vertices.max()))
                vertices = vertices * scale
                
                # Create simple monochrome colors if color data is invalid
                if len(colors) != len(faces) or colors.shape[1] != 3:
                    colors = np.ones((len(faces), 4), dtype=np.float32) * [0.7, 0.7, 0.7, 1.0]
                else:
                    # Convert RGB to RGBA
                    alpha = np.ones((len(colors), 1), dtype=np.float32)
                    colors = np.hstack([colors, alpha]).astype(np.float32)

                # Create mesh with minimal parameters
                mesh = gl.GLMeshItem(
                    vertexes=vertices,
                    faces=faces,
                    faceColors=colors,
                    smooth=False,
                    computeNormals=False,
                    drawEdges=True,
                    edgeColor=(0, 0, 0, 0.5)
                )
                
                # Add simple grid
                grid = gl.GLGridItem()
                grid.setSize(x=2, y=2, z=0.1)
                grid.setSpacing(x=0.1, y=0.1, z=0.1)
                
                # Adjust grid position
                min_z = vertices[:, 2].min()
                grid.translate(0, 0, min_z - 0.1)
                
                # Add items to viewer
                self.viewer.addItem(grid)
                self.viewer.addItem(mesh)
                
                # Set camera position
                self.viewer.setCameraPosition(distance=3.0, elevation=30, azimuth=45)
                
                # Set rendering options
                self.viewer.opts['distance'] = 3.0
                self.viewer.opts['fov'] = 60
                self.viewer.opts['elevation'] = 30
                self.viewer.opts['azimuth'] = 45
                
                # Update view
                self.viewer.update()
                
                # Show success message with detection info
                QMessageBox.information(
                    self,
                    "3D Generation Success",
                    f"Successfully created 3D model!\n\n"
                    f"Detected object type: {self.detected_object_class}\n"
                    f"Detection confidence: {self.overall_accuracy_progress.value()}%\n\n"
                    "You can now export the model or adjust the view."
                )
                
            # Re-enable convert button
            self.convert_button.setEnabled(True)
            self.convert_button.setText("🔄 Generate 3D Model")
            self.export_button.setEnabled(True)
            
        except Exception as e:
            print(f"Error displaying mesh: {str(e)}")
            QMessageBox.warning(
                self,
                "Display Error",
                "Failed to display the 3D model. Please try with a different image."
            )
            self.convert_button.setEnabled(True)
            self.convert_button.setText("🔄 Generate 3D Model")
    
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
                "STL Files (*.stl);;OBJ Files (*.obj)"
            )
            
            if file_name:
                # Add extension if not present
                if file_type == "STL Files (*.stl)" and not file_name.lower().endswith('.stl'):
                    file_name += '.stl'
                elif file_type == "OBJ Files (*.obj)" and not file_name.lower().endswith('.obj'):
                    file_name += '.obj'
                
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