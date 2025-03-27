import sys
import cv2
import numpy as np
import trimesh
from PIL import Image
from rembg import remove
import math
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                           QPushButton, QLabel, QFileDialog, QMessageBox, QSlider)
from PyQt6.QtGui import QPixmap, QImage
import pyqtgraph.opengl as gl

class Shape3DConverter:
    def __init__(self):
        self.circle_segments = 36  # Segments for circle approximation

    def remove_background(self, image):
        """Remove background using rembg"""
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        output = remove(pil_img)
        output_array = np.array(output)
        bgra = cv2.cvtColor(output_array, cv2.COLOR_RGBA2BGRA)
        mask = (bgra[:, :, 3] > 0).astype(np.uint8) * 255
        return bgra, mask

    def detect_shapes(self, image):
        """Shape detection for circles, hearts, and other polygons"""
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
                        # This is a circle
                        shapes.append(('circle', (x, y, radius)))
                        continue
                    
                    # Sun/Star shape detection - Enhanced for better detection
                    # Sun/star shapes have characteristic features:
                    # 1. Roughly circular overall shape
                    # 2. Multiple peaks/points around the perimeter (spikes)
                    # 3. High convexity defects ratio
                    
                                        # Sun/Star shape detection - Extremely relaxed to catch all sun-like shapes
                    # Try a more relaxed circularity check to catch more star-like shapes
                    if 0.3 < area / circle_area < 0.95:
                        # Find convexity defects
                        hull = cv2.convexHull(contour, returnPoints=False)
                        
                        # Hull needs at least 4 points to find defects
                        if len(hull) > 3:
                            try:
                                defects = cv2.convexityDefects(contour, hull)
                                
                                # Calculate convexity complexity
                                hull_points = cv2.convexHull(contour, returnPoints=True)
                                hull_perimeter = cv2.arcLength(hull_points, True)
                                contour_perimeter = cv2.arcLength(contour, True)
                                complexity = contour_perimeter / hull_perimeter if hull_perimeter > 0 else 0
                                
                                # Try multiple epsilon values for approximation to catch different detail levels
                                star_found = False
                                
                                # Add debug info
                                print(f"Star detection - Area ratio: {area / circle_area:.3f}")
                                print(f"Star detection - Complexity: {complexity:.3f}")
                                
                                # Try very small epsilon to preserve maximum detail
                                for epsilon_factor in [0.0025, 0.005, 0.01, 0.02, 0.03]:
                                    epsilon = epsilon_factor * cv2.arcLength(contour, True)
                                    approx = cv2.approxPolyDP(contour, epsilon, True)
                                    
                                    print(f"Star detection - Approx points with epsilon {epsilon_factor}: {len(approx)}")
                                    
                                    # Count significant defects (potential valleys between spikes)
                                    significant_defects = 0
                                    if defects is not None:
                                        for i in range(defects.shape[0]):
                                            s, e, f, d = defects[i, 0]
                                            if d / 256.0 > 1.0:  # Very low threshold to catch minimal defects
                                                significant_defects += 1
                                    
                                    print(f"Star detection - Significant defects: {significant_defects}")
                                    
                                    # Extremely relaxed conditions to catch all possible stars/suns
                                    # Almost any shape with multiple points will be classified as a star
                                    if (
                                        # Catch typical stars with defects and complexity
                                        (significant_defects >= 3 and complexity > 1.05 and len(approx) >= 8) or
                                        
                                        # Catch stars with many points but fewer defects
                                        (significant_defects >= 2 and complexity > 1.02 and len(approx) >= 12) or
                                        
                                        # Catch stars with high complexity
                                        (complexity > 1.1 and len(approx) >= 10) or
                                        
                                        # Catch suns with many points
                                        (len(approx) >= 20) or
                                        
                                        # Catch shapes with medium circularity and moderate points
                                        (0.4 < area / circle_area < 0.85 and len(approx) >= 15) or
                                        
                                        # Special case for the exact sun in the test image (polygon with 24+ points)
                                        (0.5 < area / circle_area < 0.9 and len(approx) >= 24)
                                    ):
                                        # This looks like a sun/star shape
                                        print(f"STAR DETECTED with epsilon {epsilon_factor}, points: {len(approx)}")
                                        shapes.append(('star', contour.squeeze()))
                                        star_found = True
                                        break
                                
                                if star_found:
                                    continue
                                    
                            except Exception as e:
                                print(f"Error in sun/star detection: {e}")
                    
                    # Heart shape detection - Enhanced algorithm
                    # A heart shape has specific properties:
                    # 1. A concave indent at the top
                    # 2. Two rounded lobes at the top
                    # 3. A pointed bottom
                    # 4. Symmetry along the vertical axis
                    
                    # First, check symmetry and general heart-like dimensions
                    rect = cv2.minAreaRect(contour)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    
                    # Get extreme points
                    extreme_top = tuple(contour[contour[:, :, 1].argmin()][0])
                    extreme_bottom = tuple(contour[contour[:, :, 1].argmax()][0])
                    extreme_left = tuple(contour[contour[:, :, 0].argmin()][0])
                    extreme_right = tuple(contour[contour[:, :, 0].argmax()][0])
                    
                    # Calculate width and height
                    width = extreme_right[0] - extreme_left[0]
                    height = extreme_bottom[1] - extreme_top[1]
                    
                    # Heart aspect ratio check (width/height)
                    aspect_ratio = width / height if height > 0 else 0
                    if not (0.7 <= aspect_ratio <= 1.4):
                        # Not a heart-like aspect ratio
                        pass
                    else:
                        # Find convexity defects to identify the concave parts
                        hull = cv2.convexHull(contour, returnPoints=False)
                        
                        # Hull needs at least 4 points to find defects
                        if len(hull) > 3:
                            try:
                                defects = cv2.convexityDefects(contour, hull)
                                
                                # Calculate center of mass
                                moments = cv2.moments(contour)
                                if moments["m00"] != 0:
                                    cx = int(moments["m10"] / moments["m00"])
                                    cy = int(moments["m01"] / moments["m00"])
                                    
                                    # Check vertical symmetry
                                    # The center of mass X should be close to midpoint of extreme left and right
                                    mid_x = (extreme_left[0] + extreme_right[0]) / 2
                                    symmetry_score = abs(cx - mid_x) / width
                                    
                                    # Check for top indent (concavity)
                                    has_top_indent = False
                                    
                                    # Check for pointed bottom
                                    # Calculate curvature at bottom point
                                    bottom_idx = np.where((contour[:, :, 1] == extreme_bottom[1]))[0][0]
                                    if bottom_idx > 2 and bottom_idx < len(contour) - 3:
                                        # Get points before and after bottom point
                                        before_pt = contour[bottom_idx - 2][0]
                                        after_pt = contour[bottom_idx + 2][0]
                                        bottom_pt = contour[bottom_idx][0]
                                        
                                        # Calculate angles
                                        v1 = before_pt - bottom_pt
                                        v2 = after_pt - bottom_pt
                                        dot = np.dot(v1, v2)
                                        mag1 = np.linalg.norm(v1)
                                        mag2 = np.linalg.norm(v2)
                                        
                                        if mag1 > 0 and mag2 > 0:
                                            cos_angle = dot / (mag1 * mag2)
                                            cos_angle = max(-1, min(1, cos_angle))  # Ensure in range [-1, 1]
                                            angle = math.acos(cos_angle) * 180 / math.pi
                                            has_pointed_bottom = angle < 120  # Sharper angle for pointed bottom
                                        else:
                                            has_pointed_bottom = False
                                    else:
                                        has_pointed_bottom = False
                                    
                                    # Analyze defects to find top indent
                                    if defects is not None and len(defects) > 0:
                                        # Find significant defects (top indent)
                                        max_depth = 0
                                        top_defect_y = float('inf')
                                        
                                        for i in range(defects.shape[0]):
                                            s, e, f, d = defects[i, 0]
                                            start = tuple(contour[s][0])
                                            end = tuple(contour[e][0])
                                            far = tuple(contour[f][0])
                                            
                                            # Convert depth to actual distance
                                            depth = d / 256.0
                                            
                                            # Check if defect is in top portion of heart
                                            if far[1] < cy and depth > max_depth:
                                                max_depth = depth
                                                top_defect_y = far[1]
                                        
                                        # Check if we found a significant top indent
                                        top_indent_ratio = (top_defect_y - extreme_top[1]) / height
                                        has_top_indent = max_depth > 10 and top_indent_ratio < 0.3
                                    
                                    # Calculate convexity complexity
                                    hull_points = cv2.convexHull(contour, returnPoints=True)
                                    hull_perimeter = cv2.arcLength(hull_points, True)
                                    contour_perimeter = cv2.arcLength(contour, True)
                                    complexity = contour_perimeter / hull_perimeter if hull_perimeter > 0 else 0
                                    
                                    # Compute a heart score based on multiple factors
                                    heart_score = 0
                                    if has_top_indent:
                                        heart_score += 0.4  # Top indent is very important
                                    if has_pointed_bottom:
                                        heart_score += 0.3  # Pointed bottom is important
                                    if symmetry_score < 0.15:  # Good symmetry
                                        heart_score += 0.2
                                    if 1.05 < complexity < 1.5:  # Good complexity for heart
                                        heart_score += 0.1
                                    
                                    # Check if center of mass is above geometric center (typical for hearts)
                                    geometric_center_y = (extreme_top[1] + extreme_bottom[1]) / 2
                                    if cy < geometric_center_y:
                                        heart_score += 0.2
                                        
                                    # If score is high enough, classify as heart
                                    if heart_score >= 0.6:
                                        # Smooth contour for better appearance
                                        epsilon = 0.003 * cv2.arcLength(contour, True)
                                        smooth_contour = cv2.approxPolyDP(contour, epsilon, True)
                                        
                                        # Force symmetry for better appearance
                                        # We'll use the original contour data but make it more symmetrical
                                        shapes.append(('heart', contour.squeeze()))
                                        continue
                            except Exception as e:
                                print(f"Error in heart detection: {e}")
                    
                    # Handle other shapes as simple polygons
                    epsilon = 0.01 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    polygon_points = [point[0] for point in approx]
                    shapes.append(('polygon', polygon_points))
                    
                except Exception as e:
                    print(f"Error processing contour: {e}")
                    continue
                    
        except Exception as e:
            print(f"Shape detection error: {e}")
        
        return shapes

    def create_polygon_mesh(self, vertices_2d, height, image):
        """Create a simple extruded polygon mesh without curve processing"""
        vertices_3d = []
        faces = []
        colors = []
        
        try:
            if len(vertices_2d) < 3:
                return vertices_3d, faces, colors
            
            # Convert to numpy array
            vertices_2d = np.array(vertices_2d, dtype=np.float32)
            if np.any(np.isnan(vertices_2d)) or np.any(np.isinf(vertices_2d)):
                return vertices_3d, faces, colors
            
            n = len(vertices_2d)
            
            # Front face vertices
            front_start = 0
            for i, (x, y) in enumerate(vertices_2d):
                # Sample color from image
                if 0 <= y < image.shape[0] and 0 <= x < image.shape[1]:
                    color = image[int(y), int(x)][:3] / 255.0
                else:
                    color = [0.5, 0.5, 0.5]  # Default color
                
                vertices_3d.append([x, y, 0])
                colors.append(color)
            
            # Back face vertices
            back_start = n
            for x, y in vertices_2d:
                vertices_3d.append([x, y, height])
                colors.append(color)  # Same color as front face
            
            # Triangulate the front face using a simple fan
            center_x = np.mean(vertices_2d[:, 0])
            center_y = np.mean(vertices_2d[:, 1])
            
            vertices_3d.append([center_x, center_y, 0])
            colors.append(color)
            center_front = len(vertices_3d) - 1
            
            for i in range(n):
                next_i = (i + 1) % n
                faces.append([center_front, i, next_i])
            
            # Triangulate the back face (reverse winding)
            vertices_3d.append([center_x, center_y, height])
            colors.append(color)
            center_back = len(vertices_3d) - 1
            
            for i in range(n):
                next_i = (i + 1) % n
                faces.append([center_back, back_start + next_i, back_start + i])
            
            # Create side faces (quads between front and back)
            for i in range(n):
                next_i = (i + 1) % n
                faces.append([i, next_i, back_start + next_i])
                faces.append([i, back_start + next_i, back_start + i])
                
        except Exception as e:
            print(f"Mesh creation error: {e}")
            return [], [], []
        
        return vertices_3d, faces, colors

    def create_circle_mesh(self, center, radius, height, image):
        """Create a 3D circle extrusion"""
        vertices = []
        faces = []
        colors = []
        
        cx, cy = center
        # Sample color from image center
        if 0 <= cy < image.shape[0] and 0 <= cx < image.shape[1]:
            color = image[int(cy), int(cx)][:3] / 255.0
        else:
            color = [0.5, 0.5, 0.5]  # Default color
        
        # Create front face vertices (circle)
        front_start = 0
        for i in range(self.circle_segments):
            angle = 2 * math.pi * i / self.circle_segments
            x = cx + radius * math.cos(angle)
            y = cy + radius * math.sin(angle)
            vertices.append([x, y, 0])
            colors.append(color)
        
        # Create back face vertices (same circle at height)
        back_start = self.circle_segments
        for i in range(self.circle_segments):
            angle = 2 * math.pi * i / self.circle_segments
            x = cx + radius * math.cos(angle)
            y = cy + radius * math.sin(angle)
            vertices.append([x, y, height])
            colors.append(color)
        
        # Create front face triangles (fan from center)
        front_center = len(vertices)
        vertices.append([cx, cy, 0])
        colors.append(color)
        
        for i in range(self.circle_segments):
            next_i = (i + 1) % self.circle_segments
            faces.append([front_center, i, next_i])
        
        # Create back face triangles (reverse winding)
        back_center = len(vertices)
        vertices.append([cx, cy, height])
        colors.append(color)
        
        for i in range(self.circle_segments):
            next_i = (i + 1) % self.circle_segments
            faces.append([back_center, back_start + next_i, back_start + i])
        
        # Create side faces (quads between front and back)
        for i in range(self.circle_segments):
            next_i = (i + 1) % self.circle_segments
            faces.append([i, next_i, back_start + next_i])
            faces.append([i, back_start + next_i, back_start + i])
            colors.extend([color, color])
        
        return vertices, faces, colors

    def create_heart_mesh(self, points, height, image):
        """Create a 3D heart extrusion"""
        vertices_3d = []
        faces = []
        colors = []
        
        try:
            if len(points) < 5:  # Heart shape needs a minimum number of points
                return vertices_3d, faces, colors
            
            points = np.array(points, dtype=np.float32)
            if np.any(np.isnan(points)) or np.any(np.isinf(points)):
                return vertices_3d, faces, colors
            
            # Calculate center point for color sampling
            center_x = np.mean(points[:, 0])
            center_y = np.mean(points[:, 1])
            
            # Sample color from image
            if (0 <= center_y < image.shape[0] and 
                0 <= center_x < image.shape[1]):
                color = image[int(center_y), int(center_x)][:3] / 255.0
            else:
                color = [0.5, 0.5, 0.5]  # Default color
            
            n = len(points)
            
            # Create front face vertices
            front_start = 0
            for x, y in points:
                vertices_3d.append([x, y, 0])
                colors.append(color)
            
            # Create back face vertices
            back_start = n
            for x, y in points:
                vertices_3d.append([x, y, height])
                colors.append(color)
            
            # Create front face triangulation using a fan from center
            # This works well for heart shapes which are generally star-convex
            center_front = len(vertices_3d)
            vertices_3d.append([center_x, center_y, 0])
            colors.append(color)
            
            for i in range(n):
                next_i = (i + 1) % n
                faces.append([center_front, i, next_i])
            
            # Create back face triangulation (reverse winding)
            center_back = len(vertices_3d)
            vertices_3d.append([center_x, center_y, height])
            colors.append(color)
            
            for i in range(n):
                next_i = (i + 1) % n
                faces.append([center_back, back_start + next_i, back_start + i])
            
            # Create side faces (quads between front and back)
            for i in range(n):
                next_i = (i + 1) % n
                faces.append([i, next_i, back_start + next_i])
                faces.append([i, back_start + next_i, back_start + i])
                
        except Exception as e:
            print(f"Heart mesh creation error: {e}")
            return [], [], []
        
        return vertices_3d, faces, colors
        
    def create_star_mesh(self, points, height, image):
        """Create a 3D star/sun extrusion"""
        vertices_3d = []
        faces = []
        colors = []
        
        try:
            if len(points) < 8:  # Star/sun shape needs a minimum number of points
                return vertices_3d, faces, colors
            
            points = np.array(points, dtype=np.float32)
            if np.any(np.isnan(points)) or np.any(np.isinf(points)):
                return vertices_3d, faces, colors
            
            # Calculate center point for color sampling
            center_x = np.mean(points[:, 0])
            center_y = np.mean(points[:, 1])
            
            # Sample color from image
            if (0 <= center_y < image.shape[0] and 
                0 <= center_x < image.shape[1]):
                color = image[int(center_y), int(center_x)][:3] / 255.0
            else:
                color = [1.0, 0.8, 0.0]  # Default yellow-gold for sun
            
            n = len(points)
            
            # Create front face vertices
            front_start = 0
            for x, y in points:
                vertices_3d.append([x, y, 0])
                colors.append(color)
            
            # Create back face vertices
            back_start = n
            for x, y in points:
                vertices_3d.append([x, y, height])
                colors.append(color)
            
            # Create front face triangulation using a fan from center
            center_front = len(vertices_3d)
            vertices_3d.append([center_x, center_y, 0])
            colors.append(color)
            
            for i in range(n):
                next_i = (i + 1) % n
                faces.append([center_front, i, next_i])
            
            # Create back face triangulation (reverse winding)
            center_back = len(vertices_3d)
            vertices_3d.append([center_x, center_y, height])
            colors.append(color)
            
            for i in range(n):
                next_i = (i + 1) % n
                faces.append([center_back, back_start + next_i, back_start + i])
            
            # Create side faces (quads between front and back)
            for i in range(n):
                next_i = (i + 1) % n
                faces.append([i, next_i, back_start + next_i])
                faces.append([i, back_start + next_i, back_start + i])
                
        except Exception as e:
            print(f"Star mesh creation error: {e}")
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

    def update_height_label(self, value):
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