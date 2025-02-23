import sys
import cv2
import numpy as np
import trimesh
import torch
from PIL import Image
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                           QPushButton, QLabel, QFileDialog, QMessageBox, QGroupBox, QSpinBox)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt, QThread, pyqtSignal
import pyqtgraph.opengl as gl

class SFSMeshGenerator(QThread):
    finished = pyqtSignal(np.ndarray, np.ndarray, np.ndarray)
    mesh_ready = pyqtSignal(trimesh.Trimesh)
    progress_update = pyqtSignal(str)

    def __init__(self, image, depth_scale=0.4, smoothing_iterations=2, light_direction=None):
        super().__init__()
        self.image = image
        self.depth_scale = depth_scale
        self.smoothing_iterations = smoothing_iterations
        self.light_direction = light_direction or np.array([0, 0, 1])

    def estimate_albedo(self, image):
        self.progress_update.emit("Estimating surface albedo...")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = gray.astype(float) / 255.0
        kernel = np.ones((5,5), np.float32) / 25
        local_max = cv2.dilate(gray, kernel)
        albedo = cv2.filter2D(local_max, -1, kernel)
        return albedo

    def compute_surface_normals(self, image, albedo):
        self.progress_update.emit("Computing surface normals...")
        intensity = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(float) / 255.0
        height, width = intensity.shape
        normals = np.zeros((height, width, 3))
        light = self.light_direction / np.linalg.norm(self.light_direction)
        
        for _ in range(50):
            for i in range(1, height-1):
                for j in range(1, width-1):
                    if albedo[i,j] > 0:
                        n_left = normals[i,j-1]
                        n_right = normals[i,j+1]
                        n_up = normals[i-1,j]
                        n_down = normals[i+1,j]
                        n_avg = (n_left + n_right + n_up + n_down) / 4
                        cos_theta = intensity[i,j] / (albedo[i,j] + 1e-6)
                        cos_theta = np.clip(cos_theta, -1, 1)
                        new_normal = n_avg + light * cos_theta
                        new_normal = new_normal / (np.linalg.norm(new_normal) + 1e-6)
                        normals[i,j] = new_normal
        
        return normals

    def integrate_normals(self, normals):
        self.progress_update.emit("Integrating surface normals...")
        height, width = normals.shape[:2]
        depth_map = np.zeros((height, width))
        
        for i in range(1, height):
            for j in range(1, width):
                if j > 0:
                    depth_map[i,j] = depth_map[i,j-1] - normals[i,j,0] / (normals[i,j,2] + 1e-6)
                if i > 0:
                    depth_map[i,j] += depth_map[i-1,j] - normals[i,j,1] / (normals[i,j,2] + 1e-6)
                depth_map[i,j] /= 2
        
        depth_map = cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX)
        return depth_map

    def generate_mesh(self, image):
        albedo = self.estimate_albedo(image)
        normals = self.compute_surface_normals(image, albedo)
        depth_map = self.integrate_normals(normals)
        
        self.progress_update.emit("Generating 3D mesh...")
        
        height, width = depth_map.shape
        y, x = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        
        vertices = np.zeros((height, width, 3))
        vertices[:, :, 0] = (x - width/2) / max(width, height)
        vertices[:, :, 1] = (y - height/2) / max(width, height)
        vertices[:, :, 2] = depth_map * self.depth_scale
        
        vertices = self.smooth_vertices(vertices, self.smoothing_iterations)
        
        faces = []
        colors = []
        
        for i in range(height-1):
            for j in range(width-1):
                v1 = i*width + j
                v2 = i*width + (j+1)
                v3 = (i+1)*width + j
                v4 = (i+1)*width + (j+1)
                
                faces.extend([[v1, v2, v3], [v3, v2, v4]])
                
                base_color = image[i,j] / 255.0
                shading = np.dot(normals[i,j], self.light_direction)
                shading = np.clip(shading, 0, 1)
                color = base_color * (0.3 + 0.7 * shading)
                colors.extend([color, color])
        
        vertices = vertices.reshape(-1, 3)
        faces = np.array(faces)
        colors = np.array(colors)
        
        if len(faces) > 0:
            self.progress_update.emit("Optimizing mesh...")
            mesh = trimesh.Trimesh(
                vertices=vertices,
                faces=faces,
                vertex_colors=colors.repeat(3, axis=0)
            )
            
            mesh.remove_degenerate_faces()
            mesh.remove_duplicate_faces()
            mesh.remove_unreferenced_vertices()
            mesh.fill_holes()
            mesh = mesh.smoothed()
            
            return mesh, vertices, faces, colors
        else:
            return None, vertices, np.array([]), np.array([])

    def smooth_vertices(self, vertices, iterations=2):
        self.progress_update.emit("Smoothing mesh vertices...")
        smoothed = vertices.copy()
        kernel = np.array([[0.1, 0.15, 0.1],
                          [0.15, 0.0, 0.15],
                          [0.1, 0.15, 0.1]])
        
        for _ in range(iterations):
            temp = smoothed.copy()
            for i in range(1, smoothed.shape[0] - 1):
                for j in range(1, smoothed.shape[1] - 1):
                    neighborhood = smoothed[i-1:i+2, j-1:j+2]
                    kernel_sum = kernel.sum()
                    temp[i, j] = np.sum(neighborhood * kernel[:,:,np.newaxis], axis=(0,1)) / kernel_sum
            smoothed = temp
        
        return smoothed

    def run(self):
        try:
            mesh, vertices, faces, colors = self.generate_mesh(self.image)
            if mesh is not None:
                self.progress_update.emit("Finalizing 3D model...")
                self.mesh_ready.emit(mesh)
                self.finished.emit(vertices, faces, colors)
            else:
                self.progress_update.emit("Failed to generate mesh.")
                empty_mesh = trimesh.Trimesh()
                self.mesh_ready.emit(empty_mesh)
                self.finished.emit(np.array([]), np.array([]), np.array([]))
        except Exception as e:
            self.progress_update.emit(f"Error: {str(e)}")
            empty_mesh = trimesh.Trimesh()
            self.mesh_ready.emit(empty_mesh)
            self.finished.emit(np.array([]), np.array([]), np.array([]))

import sys
import cv2
import numpy as np
import trimesh
import torch
from PIL import Image
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                           QPushButton, QLabel, QFileDialog, QMessageBox, QGroupBox, QSpinBox)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt, QThread, pyqtSignal
import pyqtgraph.opengl as gl

# [Previous SFSMeshGenerator class remains exactly the same]

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Shape from Shading 3D Generator")
        self.setGeometry(100, 100, 1200, 800)
        self.setMinimumSize(1000, 700)
        self.setup_ui_styling()
        self.init_ui()
        self.image_path = None
        self.current_mesh = None

    def setup_ui_styling(self):
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
            QSpinBox {
                background-color: #363636;
                color: white;
                border: 1px solid #404040;
                border-radius: 4px;
                padding: 2px;
            }
        """)

    def init_ui(self):
        # Create central widget and main layout
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        # Create horizontal layout for content
        content_layout = QHBoxLayout()
        
        # Left panel
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(15)
        
        # Image preview group
        image_group = QGroupBox("Image Preview")
        image_layout = QVBoxLayout(image_group)
        
        self.image_label = QLabel("No image selected")
        self.image_label.setFixedSize(400, 300)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("""
            background-color: #1e1e1e;
            border: 2px dashed #404040;
            border-radius: 4px;
            color: #808080;
        """)
        image_layout.addWidget(self.image_label)
        
        # Controls group
        controls_group = QGroupBox("Controls")
        controls_layout = QVBoxLayout(controls_group)
        
        # Depth scale control
        depth_scale_layout = QHBoxLayout()
        depth_scale_label = QLabel("Depth Scale:")
        self.depth_scale_spin = QSpinBox()
        self.depth_scale_spin.setRange(10, 100)
        self.depth_scale_spin.setValue(40)
        self.depth_scale_spin.setSuffix("%")
        depth_scale_layout.addWidget(depth_scale_label)
        depth_scale_layout.addWidget(self.depth_scale_spin)
        
        # Smoothing control
        smoothing_layout = QHBoxLayout()
        smoothing_label = QLabel("Smoothing:")
        self.smoothing_spin = QSpinBox()
        self.smoothing_spin.setRange(0, 5)
        self.smoothing_spin.setValue(2)
        smoothing_layout.addWidget(smoothing_label)
        smoothing_layout.addWidget(self.smoothing_spin)
        
        # Buttons
        self.select_button = QPushButton("📁 Select Image")
        self.select_button.setMinimumHeight(40)
        self.select_button.clicked.connect(self.select_image)
        
        self.convert_button = QPushButton("🔄 Generate 3D Model")
        self.convert_button.setMinimumHeight(40)
        self.convert_button.setEnabled(False)
        self.convert_button.clicked.connect(self.convert_to_3d)
        
        self.export_button = QPushButton("💾 Export 3D Model")
        self.export_button.setMinimumHeight(40)
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export_mesh)
        
        # Progress label
        self.progress_label = QLabel("")
        self.progress_label.setStyleSheet("color: #808080;")
        
        # Add controls to layout
        controls_layout.addLayout(depth_scale_layout)
        controls_layout.addLayout(smoothing_layout)
        controls_layout.addWidget(self.select_button)
        controls_layout.addWidget(self.convert_button)
        controls_layout.addWidget(self.export_button)
        controls_layout.addWidget(self.progress_label)
        
        # Add groups to left panel
        left_layout.addWidget(image_group)
        left_layout.addWidget(controls_group)
        left_layout.addStretch()
        
        # Right panel - 3D viewer
        viewer_group = QGroupBox("3D Preview")
        viewer_layout = QVBoxLayout(viewer_group)
        
        self.viewer = gl.GLViewWidget()
        self.viewer.setMinimumSize(600, 600)
        self.viewer.setCameraPosition(distance=40, elevation=30, azimuth=45)
        self.viewer.setBackgroundColor('#1e1e1e')
        
        # Add grid
        grid = gl.GLGridItem()
        grid.setSize(x=100, y=100, z=1)
        grid.setSpacing(x=10, y=10, z=10)
        grid.setColor((0.3, 0.3, 0.3, 1.0))
        self.viewer.addItem(grid)
        
        viewer_layout.addWidget(self.viewer)
        
        # Add panels to content layout
        content_layout.addWidget(left_panel, 1)
        content_layout.addWidget(viewer_group, 2)
        
        # Add content layout to main layout
        main_layout.addLayout(content_layout)
        
        # Set central widget
        self.setCentralWidget(central_widget)

    def select_image(self):
        """Handle image selection"""
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

    def convert_to_3d(self):
        """Convert the selected image to 3D model"""
        if not self.image_path:
            return
            
        # Load and process image
        image = cv2.imread(self.image_path)
        if image is None:
            QMessageBox.critical(
                self,
                "Error",
                "Failed to load image. Please try again with a different image."
            )
            return

        # Resize image for processing
        image = cv2.resize(image, (256, 256))  # Reduced size for performance
        
        # Start SFS mesh generation
        self.mesh_thread = SFSMeshGenerator(
            image,
            depth_scale=self.depth_scale_spin.value() / 100.0,
            smoothing_iterations=self.smoothing_spin.value()
        )
        self.mesh_thread.finished.connect(self.display_mesh)
        self.mesh_thread.mesh_ready.connect(self.store_mesh)
        self.mesh_thread.progress_update.connect(self.update_progress)
        self.mesh_thread.start()
        
        # Disable controls during processing
        self.convert_button.setEnabled(False)
        self.select_button.setEnabled(False)
        self.depth_scale_spin.setEnabled(False)
        self.smoothing_spin.setEnabled(False)

    def update_progress(self, message):
        """Update progress label with current status"""
        self.progress_label.setText(message)
        QApplication.processEvents()

    def store_mesh(self, mesh):
        """Store the generated mesh and enable export"""
        self.current_mesh = mesh
        if mesh is not None:
            self.export_button.setEnabled(True)
        
        # Re-enable controls
        self.convert_button.setEnabled(True)
        self.select_button.setEnabled(True)
        self.depth_scale_spin.setEnabled(True)
        self.smoothing_spin.setEnabled(True)

    def export_mesh(self):
        """Export the generated mesh to file"""
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
        """Display the generated mesh in the 3D viewer"""
        self.viewer.clear()
        
        if len(faces) > 0 and len(colors) > 0:
            # Create mesh visualization
            mesh = gl.GLMeshItem(
                vertexes=vertices,
                faces=faces,
                faceColors=colors,
                smooth=True,
                shader='shaded',
                drawEdges=False,
            )
            
            # Add reference grid
            grid = gl.GLGridItem()
            grid.setSize(x=100, y=100, z=1)
            grid.setSpacing(x=10, y=10, z=10)
            
            self.viewer.addItem(grid)
            self.viewer.addItem(mesh)
            
            # Adjust camera for better view
            self.viewer.setCameraPosition(
                distance=max(vertices.max() * 2, 40),
                elevation=30,
                azimuth=45
            )
            
            # Update progress
            self.progress_label.setText("3D model generation complete")
        else:
            self.progress_label.setText("Failed to generate 3D model")

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()