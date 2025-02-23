import sys
import cv2
import numpy as np
import trimesh
from PIL import Image
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                           QPushButton, QLabel, QFileDialog, QMessageBox, QGroupBox, QProgressBar)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt, QThread, pyqtSignal
import pyqtgraph.opengl as gl
import torch
import torch.nn as nn
import torchvision.transforms as transforms

class DepthEstimationNet(nn.Module):
    def __init__(self):
        super(DepthEstimationNet, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            self.conv_block(3, 32, 7),
            self.conv_block(32, 64, 5),
            self.conv_block(64, 128, 3),
            self.conv_block(128, 256, 3),
            self.conv_block(256, 512, 3),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            self.upconv_block(512, 256),
            self.upconv_block(256, 128),
            self.upconv_block(128, 64),
            self.upconv_block(64, 32),
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def conv_block(self, in_channels, out_channels, kernel_size):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        features = self.encoder(x)
        depth = self.decoder(features)
        return depth

class EnhancedMeshGenerator(QThread):
    finished = pyqtSignal(np.ndarray, np.ndarray, np.ndarray)
    mesh_ready = pyqtSignal(trimesh.Trimesh)
    progress_updated = pyqtSignal(int)

    def __init__(self, image):
        super().__init__()
        self.image = image
        self.depth_model = DepthEstimationNet()
        # Load pre-trained weights if available
        try:
            self.depth_model.load_state_dict(torch.load('depth_model.pth'))
        except:
            print("No pre-trained weights found, using initialized model")
        self.depth_model.eval()

    def estimate_depth(self, image):
        # Prepare image for depth estimation
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        
        # Convert image to PIL and apply transforms
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        input_tensor = transform(pil_image).unsqueeze(0)
        
        with torch.no_grad():
            depth_map = self.depth_model(input_tensor)
            depth_map = depth_map.squeeze().numpy()
            
        # Post-process depth map
        depth_map = cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX)
        depth_map = cv2.bilateralFilter(depth_map.astype(np.float32), 9, 75, 75)
        
        return depth_map

    def detect_objects(self, image):
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 11, 2)
        
        # Find contours with hierarchy
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                             cv2.CHAIN_APPROX_SIMPLE)
        
        # Create instance segmentation mask
        mask = np.zeros_like(gray)
        object_masks = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Filter small contours
                object_mask = np.zeros_like(gray)
                cv2.drawContours(object_mask, [contour], -1, 255, -1)
                object_masks.append(object_mask)
                cv2.drawContours(mask, [contour], -1, 255, -1)
        
        return mask, object_masks

    def generate_enhanced_mesh(self, image):
        self.progress_updated.emit(10)
        # Get depth map
        depth_map = self.estimate_depth(image)
        self.progress_updated.emit(30)
        
        # Detect objects and create masks
        shape_mask, object_masks = self.detect_objects(image)
        self.progress_updated.emit(50)
        
        # Apply mask to depth map
        masked_depth = depth_map * (shape_mask > 0).astype(float)
        
        # Generate mesh geometry
        height, width = masked_depth.shape
        y, x = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        
        # Create vertices with enhanced depth scaling
        vertices = np.zeros((height, width, 3))
        vertices[:, :, 0] = (x - width/2) / max(width, height)
        vertices[:, :, 1] = (y - height/2) / max(width, height)
        vertices[:, :, 2] = masked_depth * 0.5  # Adjusted depth scaling
        
        self.progress_updated.emit(70)
        
        # Apply advanced vertex smoothing
        vertices = self.smooth_vertices(vertices)
        
        # Generate faces and colors with object awareness
        faces = []
        colors = []
        
        for i in range(height-1):
            for j in range(width-1):
                if shape_mask[i,j] > 0:
                    # Calculate vertex indices
                    v1 = i*width + j
                    v2 = i*width + (j+1)
                    v3 = (i+1)*width + j
                    v4 = (i+1)*width + (j+1)
                    
                    # Create triangles with color information
                    faces.extend([[v1, v2, v3], [v3, v2, v4]])
                    
                    # Enhanced color calculation with depth awareness
                    base_color = image[i,j] / 255.0
                    depth_factor = masked_depth[i,j]
                    color = base_color * (0.7 + 0.3 * depth_factor)
                    colors.extend([color, color])

        self.progress_updated.emit(85)
        
        vertices = vertices.reshape(-1, 3)
        faces = np.array(faces)
        colors = np.array(colors)
        
        if len(faces) > 0:
            # Create enhanced mesh with proper topology
            mesh = trimesh.Trimesh(
                vertices=vertices,
                faces=faces,
                vertex_colors=colors.repeat(3, axis=0)
            )
            
            # Mesh optimization and enhancement
            mesh.remove_degenerate_faces()
            mesh.remove_duplicate_faces()
            mesh.remove_unreferenced_vertices()
            mesh.fill_holes()
            
            # Advanced smoothing with feature preservation
            mesh = mesh.smoothed()
            mesh = mesh.subdivide()
            mesh = mesh.smoothed()
            
            self.progress_updated.emit(100)
            return mesh, vertices, faces, colors
        else:
            self.progress_updated.emit(100)
            return None, vertices, np.array([]), np.array([])

    def smooth_vertices(self, vertices, iterations=2):
        """Apply advanced Laplacian smoothing to vertices with feature preservation"""
        smoothed = vertices.copy()
        
        for _ in range(iterations):
            temp = smoothed.copy()
            for i in range(1, smoothed.shape[0] - 1):
                for j in range(1, smoothed.shape[1] - 1):
                    # Compute weighted average of neighboring vertices
                    neighbors = np.array([
                        smoothed[i-1, j],
                        smoothed[i+1, j],
                        smoothed[i, j-1],
                        smoothed[i, j+1],
                        smoothed[i-1, j-1],
                        smoothed[i-1, j+1],
                        smoothed[i+1, j-1],
                        smoothed[i+1, j+1]
                    ])
                    
                    # Calculate weights based on distance and feature similarity
                    weights = np.exp(-np.sum((neighbors - smoothed[i,j])**2, axis=1))
                    weights /= weights.sum()
                    
                    # Apply weighted smoothing
                    temp[i, j] = np.sum(neighbors * weights[:, np.newaxis], axis=0)
            
            smoothed = temp
        
        return smoothed

    def run(self):
        try:
            mesh, vertices, faces, colors = self.generate_enhanced_mesh(self.image)
            if mesh is not None:
                self.mesh_ready.emit(mesh)
                self.finished.emit(vertices, faces, colors)
            else:
                empty_mesh = trimesh.Trimesh()
                self.mesh_ready.emit(empty_mesh)
                self.finished.emit(np.array([]), np.array([]), np.array([]))
        except Exception as e:
            print(f"Error generating mesh: {str(e)}")
            empty_mesh = trimesh.Trimesh()
            self.mesh_ready.emit(empty_mesh)
            self.finished.emit(np.array([]), np.array([]), np.array([]))

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
            QProgressBar {
                border: 2px solid #404040;
                border-radius: 5px;
                text-align: center;
                background-color: #2d2d2d;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 3px;
            }
        """)
        self.init_ui()
        self.image_path = None
        self.current_mesh = None

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
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        image_layout.addWidget(self.progress_bar)
        
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
        
        # Style for all buttons
        button_style = """
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
        
        self.select_button = QPushButton("📁 Select Image")
        self.select_button.setMinimumHeight(40)
        self.select_button.setStyleSheet(button_style)
        self.select_button.clicked.connect(self.select_image)
        
        self.convert_button = QPushButton("🔄 Generate 3D Model")
        self.convert_button.setMinimumHeight(40)
        self.convert_button.setEnabled(False)
        self.convert_button.setStyleSheet(button_style)
        self.convert_button.clicked.connect(self.convert_to_3d)

        ############################

        
        self.export_button = QPushButton("💾 Export 3D Model")
        self.export_button.setMinimumHeight(40)
        self.export_button.setEnabled(False)
        self.export_button.setStyleSheet(button_style)
        self.export_button.clicked.connect(self.export_mesh)
        
        controls_layout.addWidget(self.select_button)
        controls_layout.addWidget(self.convert_button)
        controls_layout.addWidget(self.export_button)
        
        # Add groups to left panel
        left_layout.addWidget(image_group)
        left_layout.addWidget(controls_group)
        left_layout.addStretch()
        
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
        content_layout.addWidget(left_panel, 1)
        content_layout.addWidget(viewer_group, 2)
        
        # Add content layout to main layout
        main_layout.addLayout(content_layout)
        
        self.setCentralWidget(main_widget)

    def select_image(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )
        if file_name:
            self.image_path = file_name
            pixmap = QPixmap(file_name)
            self.image_label.setPixmap(pixmap.scaled(
                self.image_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            ))
            self.convert_button.setEnabled(True)
            self.export_button.setEnabled(False)
            self.progress_bar.setVisible(False)

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def convert_to_3d(self):
        if not self.image_path:
            return
            
        # Show progress bar
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.convert_button.setEnabled(False)
        
        # Load and process image
        image = cv2.imread(self.image_path)
        image = cv2.resize(image, (256, 256))  # Reduced size for smoother results
        
        # Start enhanced mesh generation
        self.mesh_thread = EnhancedMeshGenerator(image)
        self.mesh_thread.finished.connect(self.display_mesh)
        self.mesh_thread.mesh_ready.connect(self.store_mesh)
        self.mesh_thread.progress_updated.connect(self.update_progress)
        self.mesh_thread.start()

    def store_mesh(self, mesh):
        self.current_mesh = mesh
        if mesh is not None and len(mesh.faces) > 0:
            self.export_button.setEnabled(True)
        else:
            self.export_button.setEnabled(False)
            QMessageBox.warning(
                self,
                "Warning",
                "Could not generate a valid 3D model from this image."
            )

    def export_mesh(self):
        if self.current_mesh is None:
            return
            
        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Save 3D Model",
            "",
            "STL Files (*.stl);;OBJ Files (*.obj)"
        )
        if file_name:
            try:
                self.current_mesh.export(file_name)
                QMessageBox.information(
                    self,
                    "Success",
                    f"Model exported successfully to:\n{file_name}"
                )
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error",
                    f"Export failed: {str(e)}"
                )

    def display_mesh(self, vertices, faces, colors):
        self.viewer.clear()
        
        if len(faces) > 0 and len(colors) > 0:
            # Create enhanced mesh visualization
            mesh = gl.GLMeshItem(
                vertexes=vertices,
                faces=faces,
                faceColors=colors,
                smooth=True,
                shader='shaded',
                drawEdges=False,  # Disabled edges for smoother appearance
            )
            
            # Add reference grid
            grid = gl.GLGridItem()
            grid.setSize(x=100, y=100, z=1)
            grid.setSpacing(x=10, y=10, z=10)
            grid.setColor((0.3, 0.3, 0.3, 1.0))
            
            self.viewer.addItem(grid)
            self.viewer.addItem(mesh)
            
            # Adjust camera for better view
            self.viewer.setCameraPosition(
                distance=max(vertices.max() * 2, 40),
                elevation=30,
                azimuth=45
            )
        
        # Re-enable convert button
        self.convert_button.setEnabled(True)
        self.progress_bar.setVisible(False)

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()