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

class EnhancedMeshGenerator(QThread):
    finished = pyqtSignal(np.ndarray, np.ndarray, np.ndarray)
    mesh_ready = pyqtSignal(trimesh.Trimesh)
    progress = pyqtSignal(int)

    def __init__(self, image, depth_strength=1.0, extrusion_depth=0.5, add_base=True, real_dimensions=None):
        super().__init__()
        self.image = image
        self.depth_strength = depth_strength
        self.extrusion_depth = extrusion_depth
        self.add_base = add_base
        self.real_dimensions = real_dimensions  # (width, height, depth) in mm

    def estimate_depth(self, image, contour_mask):
        """Advanced depth estimation combining multiple techniques"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Edge detection with different thresholds for better detail
        edges1 = cv2.Canny(gray, 30, 100)
        edges2 = cv2.Canny(gray, 50, 150)
        edges = cv2.addWeighted(edges1, 0.5, edges2, 0.5, 0)
        
        # Calculate structure tensor for depth cues
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        structure_tensor = np.sqrt(sobelx**2 + sobely**2)
        structure_tensor = cv2.normalize(structure_tensor, None, 0, 1, cv2.NORM_MINMAX)
        
        # Apply distance transform to get depth from shape boundaries
        dist_transform = cv2.distanceTransform(contour_mask, cv2.DIST_L2, 5)
        dist_transform = cv2.normalize(dist_transform, None, 0, 1.0, cv2.NORM_MINMAX)
        
        # Combine all depth cues with weights
        depth_map = (
            dist_transform * 0.5 +
            structure_tensor * 0.25 +
            cv2.GaussianBlur(edges.astype(float) / 255, (5, 5), 0) * 0.25
        )
        
        # Apply adaptive histogram equalization for better detail
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        depth_map = clahe.apply((depth_map * 255).astype(np.uint8)).astype(float) / 255
        
        # Bilateral filter to preserve edges while smoothing
        depth_map = cv2.bilateralFilter(depth_map.astype(np.float32), 9, 75, 75)
        
        # Apply the depth strength multiplier
        depth_map = depth_map * self.depth_strength
        
        return depth_map

    def detect_shape(self, image):
        """Enhanced shape detection focusing on the main object"""
        # If image has an alpha channel, use it as mask
        if image.shape[2] == 4:
            alpha = image[:, :, 3]
            mask = (alpha > 0).astype(np.uint8) * 255
            return mask
            
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY_INV, 11, 2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create mask for the shape
        mask = np.zeros_like(gray)
        if contours:
            # Filter contours by area to remove small noise
            min_area = (gray.shape[0] * gray.shape[1]) * 0.01  # 1% of image area
            valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
            
            if valid_contours:
                # Find the largest contour
                largest_contour = max(valid_contours, key=cv2.contourArea)
                
                # Get convex hull to ensure complete shape
                hull = cv2.convexHull(largest_contour)
                
                # Approximate the contour to get better shape detection
                epsilon = 0.02 * cv2.arcLength(hull, True)
                approx_contour = cv2.approxPolyDP(hull, epsilon, True)
                
                # Create filled mask
                cv2.drawContours(mask, [approx_contour], -1, 255, -1)
                
                # Apply morphological operations to clean up the mask
                kernel = np.ones((5,5), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask

    def generate_enhanced_mesh(self, image):
        """Generate 3D mesh with real dimensions and solid base option"""
        # Detect shape and create mask
        shape_mask = self.detect_shape(image)
        
        # Get enhanced depth map using the shape mask
        depth_map = self.estimate_depth(image, shape_mask)
        
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
        self.setWindowTitle("Enhanced 3D Object Generator")
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
                # If BGRA, convert to BGR by dropping alpha channel
                if self.processed_image.shape[2] == 4:
                    # Keep alpha channel for better shape detection
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
            
            # Check if using real dimensions
            if self.use_real_dims_checkbox.isChecked():
                real_dimensions = (
                    self.width_input.value(),
                    self.height_input.value(),
                    self.depth_input.value() / 100.0  # Scale down depth
                )
            else:
                real_dimensions = None
            
            # Downscale for better performance, but not as aggressively as before
            max_size = 256  # Larger size for better detail
            height, width = image.shape[:2]
            scale = max_size / max(height, width)
            new_size = (int(width * scale), int(height * scale))
            image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
            
            # Start mesh generation with settings
            self.mesh_thread = EnhancedMeshGenerator(
                image, 
                depth_strength=depth_strength,
                extrusion_depth=extrusion_depth,
                add_base=add_base,
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
        """Display the 3D mesh with enhanced rendering"""
        self.viewer.clear()
        
        try:
            if len(faces) > 0 and len(vertices) > 0:
                # Basic data validation
                if len(vertices) < 3 or len(faces) < 1:
                    raise ValueError("Not enough vertices or faces")

                # Ensure proper data types and shapes
                vertices = np.array(vertices, dtype=np.float32)
                faces = np.array(faces, dtype=np.uint32)
                
                # Create colors if missing
                if len(colors) != len(faces) or colors.shape[1] != 3:
                    colors = np.ones((len(faces), 4), dtype=np.float32) * [0.7, 0.7, 0.7, 1.0]
                else:
                    # Convert RGB to RGBA
                    alpha = np.ones((len(colors), 1), dtype=np.float32)
                    colors = np.hstack([colors, alpha]).astype(np.float32)

                # Create mesh with enhanced parameters
                self.mesh_item = gl.GLMeshItem(
                    vertexes=vertices,
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
                
                # Adjust grid position
                min_z = vertices[:, 2].min()
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
                
                # Set camera position
                self.viewer.setCameraPosition(distance=3.0, elevation=30, azimuth=45)
                
                # Set rendering options
                self.viewer.opts['distance'] = 3.0
                self.viewer.opts['fov'] = 60
                self.viewer.opts['elevation'] = 30
                self.viewer.opts['azimuth'] = 45
                
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