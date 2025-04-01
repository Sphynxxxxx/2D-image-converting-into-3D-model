import sys
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QFileDialog, QCheckBox)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt
import pyqtgraph.opengl as gl
import cv2

class SimpleStarGenerator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Star Generator Test")
        self.setGeometry(100, 100, 1000, 600)
        
        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)
        
        # Left panel for controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Star generation button
        self.create_button = QPushButton("Create 3D Star")
        self.create_button.clicked.connect(self.create_star)
        
        # Option for classic vs 3D star
        self.true_3d_checkbox = QCheckBox("True 3D Star")
        self.true_3d_checkbox.setChecked(True)
        
        # Add to left layout
        left_layout.addWidget(self.create_button)
        left_layout.addWidget(self.true_3d_checkbox)
        left_layout.addStretch()
        
        # Right panel for 3D view
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # 3D viewer
        self.viewer = gl.GLViewWidget()
        self.viewer.setCameraPosition(distance=5, elevation=30, azimuth=45)
        self.viewer.setBackgroundColor('black')
        
        # Add grid for reference
        grid = gl.GLGridItem()
        grid.setSize(4, 4, 4)
        grid.setSpacing(1, 1, 1)
        self.viewer.addItem(grid)
        
        # Add to right layout
        right_layout.addWidget(self.viewer)
        
        # Add panels to main layout
        layout.addWidget(left_panel, 1)
        layout.addWidget(right_panel, 3)
    
    def create_star(self):
        """Create a 3D star"""
        # Clear previous content
        self.viewer.clear()
        
        # Add grid back
        grid = gl.GLGridItem()
        grid.setSize(4, 4, 4)
        grid.setSpacing(1, 1, 1)
        self.viewer.addItem(grid)
        
        # Parameters
        num_points = 5
        outer_radius = 2.0
        inner_radius = outer_radius * 0.4
        height = 1.0
        true_3d = self.true_3d_checkbox.isChecked()
        
        # Create vertices
        vertices = []
        
        # Center point (origin)
        vertices.append([0, 0, 0])
        
        # Create points around the star (on XY plane)
        for i in range(num_points * 2):
            angle = 2 * np.pi * i / (num_points * 2)
            radius = inner_radius if i % 2 else outer_radius
            
            x = radius * np.cos(angle - np.pi/2)
            y = radius * np.sin(angle - np.pi/2)
            
            vertices.append([x, y, 0])
        
        # For true 3D star, add apex points
        if true_3d:
            # Add top and bottom apex points
            front_apex_idx = len(vertices)
            vertices.append([0, 0, height])  # Top apex
            back_apex_idx = len(vertices)
            vertices.append([0, 0, -height/2])  # Bottom apex
            
            # Create the faces array
            faces = []
            
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
                
                x = radius * np.cos(angle - np.pi/2)
                y = radius * np.sin(angle - np.pi/2)
                
                vertices.append([x, y, height])
            
            # Create the faces array
            faces = []
            
            # Add center vertices (front and back)
            front_center = len(vertices)
            vertices.append([0, 0, 0])
            back_center = len(vertices)
            vertices.append([0, 0, height])
            
            # Front faces (connecting to front center)
            for i in range(num_points * 2):
                next_i = (i + 1) % (num_points * 2)
                faces.append([front_center, i + 1, next_i + 1])
            
            # Back faces (connecting to back center)
            for i in range(num_points * 2):
                next_i = (i + 1) % (num_points * 2)
                # Note reversed winding order
                faces.append([back_center, back_start + next_i, back_start + i])
            
            # Side faces
            for i in range(num_points * 2):
                next_i = (i + 1) % (num_points * 2)
                # First triangle
                faces.append([i + 1, next_i + 1, back_start + next_i])
                # Second triangle
                faces.append([i + 1, back_start + next_i, back_start + i])
        
        vertices = np.array(vertices, dtype=np.float32)
        faces = np.array(faces, dtype=np.uint32)
        
        # Create colors
        colors = []
        for i in range(len(vertices)):
            # Alternate colors for vertices to make the star more visible
            if i % 2 == 0:
                colors.append([1.0, 0.5, 0.0, 1.0])  # Orange
            else:
                colors.append([1.0, 0.8, 0.0, 1.0])  # Yellow
        
        # Create mesh item
        star_mesh = gl.GLMeshItem(
            vertexes=vertices,
            faces=faces,
            faceColors=colors,
            smooth=True,
            drawEdges=True,
            edgeColor=(0.5, 0.5, 0.5, 1),
            shader='shaded'
        )
        
        # Add to viewer
        self.viewer.addItem(star_mesh)
        
        # Add axis for reference
        x_axis = gl.GLLinePlotItem(pos=np.array([[0,0,0], [2,0,0]]), color=(1,0,0,1), width=2)
        y_axis = gl.GLLinePlotItem(pos=np.array([[0,0,0], [0,2,0]]), color=(0,1,0,1), width=2)
        z_axis = gl.GLLinePlotItem(pos=np.array([[0,0,0], [0,0,2]]), color=(0,0,1,1), width=2)
        
        self.viewer.addItem(x_axis)
        self.viewer.addItem(y_axis)
        self.viewer.addItem(z_axis)
        
        # Reset camera
        self.viewer.setCameraPosition(distance=outer_radius*3, elevation=30, azimuth=45)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SimpleStarGenerator()
    window.show()
    sys.exit(app.exec())