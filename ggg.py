import sys
import numpy as np
import cv2
import trimesh
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                           QPushButton, QLabel, QFileDialog, QSlider, QGroupBox, QCheckBox,
                           QComboBox, QMessageBox, QProgressBar)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt, QThread, pyqtSignal
import pyqtgraph.opengl as gl
from PIL import Image
from rembg import remove
import math
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

class HeartNet(nn.Module):
    """Simple CNN for heart shape verification"""
    def __init__(self):
        super(HeartNet, self).__init__()
        self.conv1 = nn.Conv2d(4, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 1)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

class AIHeartProcessor:
    """AI-based heart processing with pre-trained weights"""
    def __init__(self):
        # Mock AI model (would load pre-trained weights in real implementation)
        self.heart_model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def init_model(self):
        """Initialize the heart detection model"""
        if self.heart_model is None:
            self.heart_model = HeartNet()
            # In a real implementation: self.heart_model.load_state_dict(torch.load("heart_model.pth"))
            self.heart_model.to(self.device)
            self.heart_model.eval()
        
    def is_heart_shape(self, image, contour):
        """Use AI to verify if contour is a heart shape"""
        # This is simplified and would use the actual model in a real implementation
        # Here we do basic shape analysis instead
        
        # Create a mask from the contour
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        
        # Get aspect ratio
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        
        # Get convexity and check center of mass
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        contour_area = cv2.contourArea(contour)
        
        solidity = float(contour_area)/hull_area if hull_area > 0 else 0
        
        # Get moments and center of mass
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # Check if center of mass is in upper half
            com_in_upper_half = cy < (y + h/2)
        else:
            com_in_upper_half = False
            
        # Heart criteria
        is_heart = (
            0.8 < aspect_ratio < 1.2 and  # Width/height ratio
            solidity > 0.8 and           # Relatively convex
            com_in_upper_half            # Center of mass in upper half
        )
        
        return is_heart, 0.95 if is_heart else 0.1  # Confidence score
        
    def optimize_heart_shape(self, contour, confidence):
        """Improve heart contour based on ideal heart proportions"""
        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)
        center_x = x + w/2
        center_y = y + h/2
        
        # If confidence is low, replace with ideal heart shape
        if confidence < 0.7:
            return self.create_ideal_heart_contour((center_x, center_y), max(w, h)/2)
        
        # Get convex hull
        hull = cv2.convexHull(contour)
        
        # Find top center point (dip between lobes)
        top_points = contour[contour[:, :, 1] == np.min(contour[:, :, 1])]
        top_center_x = np.mean(top_points[:, 0])
        
        # Find bottom point
        bottom_points = contour[contour[:, :, 1] == np.max(contour[:, :, 1])]
        bottom_x = np.mean(bottom_points[:, 0])
        
        # Check if top center dip and bottom point are aligned
        if abs(top_center_x - bottom_x) > w*0.2:
            # Not well aligned, use ideal heart
            return self.create_ideal_heart_contour((center_x, center_y), max(w, h)/2)
            
        return contour
    
    def create_ideal_heart_contour(self, center, size):
        """Create an ideal heart contour"""
        cx, cy = center
        points = []
        
        # Create a heart using the parametric equations
        for i in range(100):
            t = 2 * math.pi * i / 100
            x = cx + size * (16 * math.pow(math.sin(t), 3)) / 16
            y = cy + size * (13 * math.cos(t) - 5 * math.cos(2*t) - 2 * math.cos(3*t) - math.cos(4*t)) / 16
            points.append([int(x), int(y)])
            
        return np.array(points, dtype=np.int32).reshape((-1, 1, 2))


class AdvancedHeart3DConverter:
    def __init__(self):
        self.ai_processor = AIHeartProcessor()
        self.segments_horizontal = 96  # Controls smoothness
        self.segments_vertical = 64
        self.model_type = "anatomical"  # Default model type
        
    def remove_background(self, image):
        """Remove background using rembg"""
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        output = remove(pil_img)
        output_array = np.array(output)
        bgra = cv2.cvtColor(output_array, cv2.COLOR_RGBA2BGRA)
        mask = (bgra[:, :, 3] > 0).astype(np.uint8) * 255
        return bgra, mask
    
    def detect_heart(self, image):
        """AI-enhanced heart detection"""
        # Initialize AI model
        self.ai_processor.init_model()
        
        # Convert to grayscale
        if image.shape[2] == 4:
            # Image with alpha channel
            gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Threshold
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find largest contour
        if not contours:
            return None, None, 0
            
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Check if contour is large enough
        if cv2.contourArea(largest_contour) < 100:
            return None, None, 0
            
        # Use AI to verify and optimize heart shape
        is_heart, confidence = self.ai_processor.is_heart_shape(image, largest_contour)
        
        if is_heart:
            # Optimize the heart shape
            optimized_contour = self.ai_processor.optimize_heart_shape(largest_contour, confidence)
            
            # Get dominant color
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [optimized_contour], -1, 255, -1)
            
            # Extract color
            if image.shape[2] == 4:  # BGRA
                mean_color = cv2.mean(image, mask=mask)
                color = [mean_color[2]/255.0, mean_color[1]/255.0, mean_color[0]/255.0, 1.0]  # RGBA
            else:  # BGR
                mean_color = cv2.mean(image, mask=mask)
                color = [mean_color[2]/255.0, mean_color[1]/255.0, mean_color[0]/255.0, 1.0]  # RGBA
                
            return optimized_contour.squeeze(), color, confidence
            
        return None, None, 0
    
    def create_anatomical_heart(self, center, size, height, color):
        """Create anatomically-inspired 3D heart"""
        vertices = []
        faces = []
        colors = []
        
        cx, cy = center
        
        # Anatomical heart parameters
        atrium_size = 0.6  # Size of upper chambers
        ventricle_size = 0.9  # Size of lower chambers
        indent_depth = 0.5  # Depth of indent between chambers
        
        # Generate heart chambers using parametric equations
        for v_idx in range(self.segments_vertical + 1):
            v = v_idx / self.segments_vertical
            phi = v * math.pi  # 0 to π
            
            for h_idx in range(self.segments_horizontal):
                u = 2 * math.pi * h_idx / self.segments_horizontal
                
                # Basic values
                sin_phi = math.sin(phi)
                cos_phi = math.cos(phi)
                sin_u = math.sin(u)
                cos_u = math.cos(u)
                
                # Heart shape parameters - more anatomically correct
                width_factor = 1.0
                depth_factor = 1.0
                
                # Divide heart into regions (atria and ventricles)
                if phi < math.pi * 0.35:  # Atria (upper chambers)
                    # Left and right atria - bulge on upper sides
                    atria_factor = math.sin(2 * u) * 0.8
                    width_factor = atrium_size * (1.0 + atria_factor * (1.0 - phi/(math.pi * 0.35)))
                    
                    # Indent between atria
                    if abs(u - math.pi) < 0.6:
                        center_dip = indent_depth * (1.0 - abs(u - math.pi) / 0.6)
                        width_factor -= center_dip * (1.0 - phi/(math.pi * 0.35))
                    
                    # Depth is less in upper chambers
                    depth_factor = 0.7 * (1.0 + 0.5 * phi/(math.pi * 0.35))
                
                elif phi < math.pi * 0.7:  # Transition zone and upper ventricles
                    # Smooth transition from atria to ventricles
                    mid_region = (phi - math.pi * 0.35) / (math.pi * 0.35)
                    width_factor = atrium_size * (1.0 - mid_region) + ventricle_size * mid_region
                    
                    # Side bulges for ventricles (wider in middle)
                    ventricle_bulge = 0.3 * math.sin(u - math.pi/2) * mid_region
                    width_factor += ventricle_bulge
                    
                    # Transition to deeper in ventricles
                    depth_factor = 0.9 + 0.7 * mid_region
                    
                    # Keep the indent
                    if abs(u - math.pi) < 0.4:
                        center_indent = 0.3 * (1.0 - abs(u - math.pi) / 0.4) * (1.0 - mid_region)
                        width_factor -= center_indent
                
                else:  # Ventricles and apex (bottom)
                    # Taper to apex (bottom point)
                    bottom_taper = 1.0 - 0.9 * ((phi - math.pi * 0.7) / (math.pi * 0.3)) ** 1.2
                    width_factor = ventricle_size * bottom_taper
                    
                    # Maintain asymmetry (left ventricle larger than right)
                    if u < math.pi:
                        width_factor *= 1.1  # Left ventricle is larger
                        
                    # Depth gets narrower toward apex
                    depth_factor = 1.6 * (1.0 - 0.7 * ((phi - math.pi * 0.7) / (math.pi * 0.3)))
                
                # Add surface details and vascular impressions
                detail_factor = 0.05 * math.sin(6 * u) * math.sin(8 * phi)
                width_factor += detail_factor * height  # Scale detail with height
                
                # Asymmetry - shift the heart slightly
                u_shifted = u
                if phi > math.pi * 0.5:
                    u_shifted = u + (phi - math.pi * 0.5) * 0.1 * math.sin(u)
                
                # Calculate coordinates with all factors
                x = cx + size * width_factor * sin_phi * math.cos(u_shifted)
                y = cy + size * width_factor * sin_phi * sin_u
                
                # Create more anatomically correct depth profile
                depth_multiplier = 2.0 * height
                z = size * depth_factor * cos_phi * depth_multiplier
                
                # Add a slight twist to the heart
                angle = 0.1 * phi
                x_rot = x * math.cos(angle) - z * math.sin(angle)
                z_rot = x * math.sin(angle) + z * math.cos(angle)
                
                vertices.append([x_rot, y, z_rot])
                colors.append(color)
        
        # Generate face triangles
        for v_idx in range(self.segments_vertical):
            for h_idx in range(self.segments_horizontal):
                # Current row indices
                curr1 = v_idx * self.segments_horizontal + h_idx
                curr2 = v_idx * self.segments_horizontal + (h_idx + 1) % self.segments_horizontal
                
                # Next row indices
                next1 = (v_idx + 1) * self.segments_horizontal + h_idx
                next2 = (v_idx + 1) * self.segments_horizontal + (h_idx + 1) % self.segments_horizontal
                
                # Create triangles
                faces.append([curr1, curr2, next2])
                faces.append([curr1, next2, next1])
        
        # Create mesh
        mesh = trimesh.Trimesh(
            vertices=np.array(vertices),
            faces=np.array(faces),
            vertex_colors=np.array(colors)
        )
        
        return mesh
    
    def create_stylized_heart(self, center, size, height, color):
        """Create a stylized iconic heart shape with proper lobes and indent"""
        vertices = []
        faces = []
        colors = []
        
        # Use a rich deep red color for a glossy look
        heart_color = [0.7, 0.05, 0.05, 1.0]  # Deep red
        if color is not None:
            heart_color = color
        
        cx, cy = center
        
        # Parameters for controlling heart shape
        lobe_depth = 0.3  # Controls the indent between lobes
        lobe_width = 1.1  # Controls width of lobes
        bottom_taper = 1.2  # Controls sharpness of bottom point
        
        # Generate heart using enhanced parametric equations
        for v_idx in range(self.segments_vertical + 1):
            v = v_idx / self.segments_vertical
            phi = v * math.pi  # 0 to π - angle around vertical axis
            
            for h_idx in range(self.segments_horizontal):
                u = 2 * math.pi * h_idx / self.segments_horizontal  # 0 to 2π - angle around horizontal plane
                
                # Calculate base heart shape
                sint = math.sin(u)
                cost = math.cos(u)
                
                # Classic heart curve formula with enhancements
                x_base = 16 * math.pow(sint, 3)
                y_base = 13 * cost - 5 * math.cos(2*u) - 2 * math.cos(3*u) - math.cos(4*u)
                
                # Scale and center
                x = cx + size * x_base / 17  # Normalize to keep proportional
                y = cy + size * y_base / 17
                
                # Enhanced indent between lobes - deepen the center valley
                if abs(u - math.pi) < 0.5:  # Near top center
                    center_factor = 1.0 - 2.0 * abs(u - math.pi)
                    x -= size * lobe_depth * center_factor * math.pow(1.0 - phi, 2)
                
                # Enhance lobe width
                if u < math.pi/2 or u > 3*math.pi/2:  # Side lobes
                    x *= lobe_width
                
                # Calculate z-coordinate for 3D volume
                # Base z on distance from center for fuller shape
                dist_from_center = math.sqrt((x-cx)**2 + (y-cy)**2)
                max_dist = size * 1.5
                normalized_dist = min(dist_from_center / max_dist, 1.0)
                
                # Volume profile - fuller in the middle, tapered at edges
                volume_factor = math.pow(math.sin(phi), 0.8)  # Adjust power for roundness
                edge_taper = 1.0 - 0.7 * math.pow(normalized_dist, 1.5)
                z = height * size * volume_factor * edge_taper
                
                # Add more volume to the lobes
                if phi < math.pi * 0.3:  # Top part (lobes)
                    if abs(sint) > 0.3:  # Side lobes
                        z *= 1.2  # More volume in lobes
                
                # Add glossy highlights effect through subtle surface variation
                surface_detail = 0.02 * size * math.sin(8*u) * math.sin(6*phi)
                z += surface_detail
                
                vertices.append([x, y, z])
                colors.append(heart_color)
        
        # Generate faces (same as before)
        for v_idx in range(self.segments_vertical):
            for h_idx in range(self.segments_horizontal):
                curr1 = v_idx * self.segments_horizontal + h_idx
                curr2 = v_idx * self.segments_horizontal + (h_idx + 1) % self.segments_horizontal
                next1 = (v_idx + 1) * self.segments_horizontal + h_idx
                next2 = (v_idx + 1) * self.segments_horizontal + (h_idx + 1) % self.segments_horizontal
                
                faces.append([curr1, curr2, next2])
                faces.append([curr1, next2, next1])
        
        # Create mesh
        mesh = trimesh.Trimesh(
            vertices=np.array(vertices),
            faces=np.array(faces),
            vertex_colors=np.array(colors)
        )
        
        return mesh
    
    def create_3d_heart(self, contour, height, color, model_type="anatomical"):
        """Create 3D heart mesh based on selected model type"""
        try:
            # Get center and dimensions
            x, y, w, h = cv2.boundingRect(contour)
            center_x = x + w/2
            center_y = y + h/2
            radius = max(w, h) / 2
            
            # Create 3D heart based on selected model type
            if model_type == "anatomical":
                return self.create_anatomical_heart((center_x, center_y), radius, height, color)
            elif model_type == "stylized":
                return self.create_stylized_heart((center_x, center_y), radius, height, color)
            elif model_type == "extrusion":
                # Standard extrusion of the contour
                return self.create_extruded_heart(contour, height, color)
            elif model_type == "rounded":
                # Use the rounded heart function
                return self.create_balloon_heart((center_x, center_y), radius, height, color)
            else:
                return self.create_anatomical_heart((center_x, center_y), radius, height, color)
                    
        except Exception as e:
            print(f"Error creating 3D heart: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    
    def create_extruded_heart(self, contour, height, color):
        """Create a simple extrusion of the heart contour"""
        vertices = []
        faces = []
        colors = []
        
        try:
            # Resample to fixed point count for better quality
            points = self.resample_contour(contour, 100)
            
            # Get center
            center_x = np.mean(points[:, 0])
            center_y = np.mean(points[:, 1])
            
            # Create front and back faces
            n_points = len(points)
            
            # Front face vertices
            for x, y in points:
                vertices.append([x, y, 0])
                colors.append(color)
            
            # Back face vertices
            for x, y in points:
                vertices.append([x, y, height * 100])  # Scale height
                colors.append(color)
            
            # Front center for triangulation
            front_center = len(vertices)
            vertices.append([center_x, center_y, 0])
            colors.append(color)
            
            # Front face triangles
            for i in range(n_points):
                faces.append([front_center, i, (i+1)%n_points])
            
            # Back center for triangulation
            back_center = len(vertices)
            vertices.append([center_x, center_y, height * 100])
            colors.append(color)
            
            # Back face triangles
            for i in range(n_points):
                faces.append([back_center, n_points+(i+1)%n_points, n_points+i])
            
            # Side faces
            for i in range(n_points):
                j = (i+1) % n_points
                faces.append([i, j, n_points+j])
                faces.append([i, n_points+j, n_points+i])
            
            # Create the mesh
            mesh = trimesh.Trimesh(
                vertices=np.array(vertices),
                faces=np.array(faces),
                vertex_colors=np.array(colors)
            )
            
            return mesh
            
        except Exception as e:
            print(f"Error in extruded heart: {e}")
            return None
    
    def resample_contour(self, contour, num_points):
        """Resample contour to have exactly num_points points"""
        # Flatten contour if needed
        if len(contour.shape) > 2:
            contour = np.squeeze(contour)
            
        if len(contour) == num_points:
            return contour
            
        # Calculate cumulative distance
        contour_length = cv2.arcLength(np.array([contour], dtype=np.float32), True)
        step_length = contour_length / num_points
        
        # Create new contour
        new_contour = np.zeros((num_points, 2), dtype=np.float32)
        
        # Add first point
        new_contour[0] = contour[0]
        
        # Add remaining points
        point_idx = 1
        dist_accumulated = 0
        
        for i in range(1, len(contour)):
            # Calculate distance from previous point
            dist = np.linalg.norm(contour[i] - contour[i-1])
            
            # Add points at regular intervals
            while dist_accumulated + dist >= step_length and point_idx < num_points:
                # Interpolate to get exact position
                fraction = (step_length - dist_accumulated) / dist
                new_point = contour[i-1] + fraction * (contour[i] - contour[i-1])
                
                # Add the point
                new_contour[point_idx] = new_point
                point_idx += 1
                
                # Update accumulated distance
                dist_accumulated = 0
                
                # Adjust remaining distance
                dist = (1 - fraction) * dist
            
            # Update accumulated distance
            dist_accumulated += dist
            
        # Fill any remaining points
        while point_idx < num_points:
            new_contour[point_idx] = contour[-1]
            point_idx += 1
            
        return new_contour
    
    def create_balloon_heart(self, center, size, height, color):
        """Create a completely inflated balloon heart shape"""
        vertices = []
        faces = []
        colors = []
        
        cx, cy = center
        
        # Use a bright balloon-red color
        heart_color = [0.95, 0.15, 0.15, 1.0]  # Bright balloon red
        if color is not None:
            heart_color = color
        
        # Use spherical coordinates approach for complete inflation
        u_segments = 64  # Horizontal resolution
        v_segments = 64  # Vertical resolution
        
        # First, generate a base heart curve for reference
        heart_points = []
        for t in range(100):
            angle = 2 * math.pi * t / 100
            x = 16 * math.pow(math.sin(angle), 3)
            y = 13 * math.cos(angle) - 5 * math.cos(2*angle) - 2 * math.cos(3*angle) - math.cos(4*angle)
            heart_points.append((x/17.0, y/17.0))  # Normalize to approximately -1 to 1 range
        
        # Build the 3D heart by sweeping the heart curve through spherical coordinates
        for v_idx in range(v_segments + 1):
            # Vertical angle from 0 to pi (top to bottom)
            phi = v_idx / v_segments * math.pi
            sin_phi = math.sin(phi)
            cos_phi = math.cos(phi)
            
            for u_idx in range(u_segments):
                # Horizontal angle from 0 to 2pi (around the heart)
                theta = u_idx / u_segments * 2 * math.pi
                sin_theta = math.sin(theta)
                cos_theta = math.cos(theta)
                
                # Sample the heart curve based on theta
                sample_idx = int((theta / (2 * math.pi)) * len(heart_points))
                sample_idx = min(sample_idx, len(heart_points) - 1)
                heart_x, heart_y = heart_points[sample_idx]
                
                # Base radius determined by heart curve
                base_radius = math.sqrt(heart_x**2 + heart_y**2) * size
                
                # Full 3D spherical coordinates with heart-shape modulation
                # The sin_phi ensures a proper sphere-like shape that's closed at top and bottom
                x = cx + base_radius * sin_phi * cos_theta
                y = cy + base_radius * sin_phi * sin_theta
                
                # Distort z based on heart shape
                heart_factor = (heart_y + 0.5) * 1.5  # Transform to positive range and scale
                z = base_radius * cos_phi * heart_factor * height
                
                # Adjust for heart orientation
                x_final = x
                y_final = y * 0.8 + z * 0.2  # Blend y and z for proper heart orientation
                z_final = z * 0.8 - y * 0.2  # Blend z and y
                
                vertices.append([x_final, y_final, z_final])
                colors.append(heart_color)
        
        # Generate faces - connect adjacent vertices
        for v_idx in range(v_segments):
            for u_idx in range(u_segments):
                # Current indices
                i0 = v_idx * u_segments + u_idx
                i1 = v_idx * u_segments + (u_idx + 1) % u_segments
                i2 = (v_idx + 1) * u_segments + u_idx
                i3 = (v_idx + 1) * u_segments + (u_idx + 1) % u_segments
                
                # Create triangular faces
                faces.append([i0, i1, i3])
                faces.append([i0, i3, i2])
        
        # Create mesh
        mesh = trimesh.Trimesh(
            vertices=np.array(vertices),
            faces=np.array(faces),
            vertex_colors=np.array(colors)
        )
        
        # Add slight smoothing if needed
        if hasattr(self, 'smooth_checkbox') and self.smooth_checkbox.isChecked():
            try:
                # Use gentle taubin smoothing instead of laplacian 
                lamb = 0.5    # Positive smoothing factor
                mu = -0.53    # Negative smoothing factor
                iterations = 5
                
                # Apply Taubin smoothing
                new_vertices = np.array(mesh.vertices, dtype=np.float64)
                adjacency = mesh.vertex_neighbors
                
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
                    
                mesh.vertices = new_vertices
            except Exception as e:
                print(f"Error during smoothing: {e}")
        
        return mesh

class ProcessThread(QThread):
    """Background processing thread to keep UI responsive"""
    update_progress = pyqtSignal(int)
    processing_done = pyqtSignal(object)
    
    def __init__(self, converter, contour, height, color, model_type):
        super().__init__()
        self.converter = converter
        self.contour = contour
        self.height = height
        self.color = color
        self.model_type = model_type
        
    def run(self):
        # Simulate progress
        for i in range(0, 101, 5):
            self.update_progress.emit(i)
            time.sleep(0.05)
        
        # Create the 3D model
        mesh = self.converter.create_3d_heart(
            self.contour,
            self.height,
            self.color,
            self.model_type
        )
        
        # Emit result
        self.processing_done.emit(mesh)

        


class AdvancedHeartConverterApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.converter = AdvancedHeart3DConverter()
        self.original_image = None
        self.processed_image = None
        self.heart_contour = None
        self.heart_color = None
        self.heart_confidence = 0
        self.current_mesh = None
        
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle("AI Heart 3D Converter")
        self.setGeometry(100, 100, 1000, 700)
        
        # Main layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        
        # Left panel for controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Image display
        self.image_label = QLabel("No image loaded")
        self.image_label.setFixedSize(400, 400)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("QLabel { background-color: #f0f0f0; border: 2px dashed #aaa; }")
        left_layout.addWidget(self.image_label)
        
        # Image controls
        image_group = QGroupBox("Image Controls")
        image_layout = QVBoxLayout()
        
        self.load_button = QPushButton("Load Heart Image")
        self.load_button.clicked.connect(self.load_image)
        
        self.remove_bg_button = QPushButton("Remove Background")
        self.remove_bg_button.setEnabled(False)
        self.remove_bg_button.clicked.connect(self.remove_background)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        
        # Heart model selector
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Heart Model:"))
        self.model_selector = QComboBox()
        self.model_selector.addItems(["Anatomical Heart", "Stylized Heart", "Simple Extrusion", "Rounded Heart"])
        model_layout.addWidget(self.model_selector)
        
        # Convert button
        self.convert_button = QPushButton("Convert to 3D Heart")
        self.convert_button.setEnabled(False)
        self.convert_button.clicked.connect(self.convert_to_3d)
        self.convert_button.setStyleSheet("QPushButton { background-color: #ff9999; font-weight: bold; }")
        
        self.export_button = QPushButton("Export 3D Model")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export_model)
        
        image_layout.addWidget(self.load_button)
        image_layout.addWidget(self.remove_bg_button)
        image_layout.addLayout(model_layout)
        image_layout.addWidget(self.convert_button)
        image_layout.addWidget(self.progress_bar)
        image_layout.addWidget(self.export_button)
        image_group.setLayout(image_layout)
        left_layout.addWidget(image_group)
        
        # Heart options
        options_group = QGroupBox("Heart Options")
        options_layout = QVBoxLayout()
        
        # Depth/volume slider
        depth_layout = QVBoxLayout()
        self.depth_label = QLabel("Heart Size: 0.5")
        self.depth_slider = QSlider(Qt.Orientation.Horizontal)
        self.depth_slider.setRange(10, 200)
        self.depth_slider.setValue(50)
        self.depth_slider.valueChanged.connect(self.update_depth)
        
        depth_layout.addWidget(QLabel("Size/Volume:"))
        depth_layout.addWidget(self.depth_slider)
        depth_layout.addWidget(self.depth_label)
        options_layout.addLayout(depth_layout)
        
        # Rendering options
        render_options = QHBoxLayout()
        self.smooth_checkbox = QCheckBox("Smooth Surface")
        self.smooth_checkbox.setChecked(True)
        
        self.detail_checkbox = QCheckBox("High Detail")
        self.detail_checkbox.setChecked(True)
        self.detail_checkbox.stateChanged.connect(self.update_detail_level)
        
        self.shade_checkbox = QCheckBox("Realistic Shading")
        self.shade_checkbox.setChecked(True)
        self.shade_checkbox.stateChanged.connect(self.update_shading)
        
        render_options.addWidget(self.smooth_checkbox)
        render_options.addWidget(self.detail_checkbox)
        render_options.addWidget(self.shade_checkbox)
        options_layout.addLayout(render_options)
        
        # Confidence display
        self.confidence_label = QLabel("Heart Detection: No heart detected")
        options_layout.addWidget(self.confidence_label)
        
        options_group.setLayout(options_layout)
        left_layout.addWidget(options_group)
        
        left_layout.addStretch()
        
        # Right panel for 3D viewer
        self.viewer = gl.GLViewWidget()
        self.viewer.setCameraPosition(distance=5)
        
        # Add axes grid
        grid = gl.GLGridItem()
        grid.setSize(2, 2)
        self.viewer.addItem(grid)
        
        # Add panels to main layout
        main_layout.addWidget(left_panel, 1)
        main_layout.addWidget(self.viewer, 2)
    
    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open Heart Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )
        
        if file_name:
            self.original_image = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)
            
            # Convert to BGRA if needed
            if self.original_image.shape[2] == 3:
                self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2BGRA)
                
            # Display the image
            self.display_image(self.original_image)
            
            # Enable buttons
            self.remove_bg_button.setEnabled(True)
            self.convert_button.setEnabled(True)

    def remove_background(self):
                
        if self.original_image is not None:
            try:
                # Show progress indicator
                self.progress_bar.setVisible(True)
                self.progress_bar.setValue(10)
                
                # Remove background
                result, _ = self.converter.remove_background(self.original_image)
                self.processed_image = result
                
                self.progress_bar.setValue(80)
                
                # Display processed image
                self.display_image(self.processed_image)
                
                # Reset heart detection
                self.heart_contour = None
                self.heart_color = None
                self.heart_confidence = 0
                self.confidence_label.setText("Heart Detection: No heart detected")
                
                self.progress_bar.setValue(100)
                
                # Auto-detect heart
                self.detect_heart()
                
            except Exception as e:
                print(f"Error removing background: {e}")
                QMessageBox.warning(self, "Error", f"Failed to remove background: {str(e)}")
                
            finally:
                self.progress_bar.setVisible(False)
    
    def detect_heart(self):
        """Detect heart shape in the image"""
        image = self.processed_image if self.processed_image is not None else self.original_image
        
        if image is None:
            return
            
        try:
            # Set progress indicator
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(20)
            
            # Detect heart
            contour, color, confidence = self.converter.detect_heart(image)
            
            self.progress_bar.setValue(90)
            
            if contour is not None and color is not None:
                self.heart_contour = contour
                self.heart_color = color
                self.heart_confidence = confidence
                
                # Update confidence display
                confidence_pct = int(confidence * 100)
                status = "Excellent" if confidence_pct > 90 else "Good" if confidence_pct > 70 else "Fair"
                self.confidence_label.setText(f"Heart Detection: {status} ({confidence_pct}%)")
                
                # If confidence is good, highlight the detected heart
                self.highlight_heart_contour()
            else:
                self.confidence_label.setText("Heart Detection: No heart detected")
                QMessageBox.information(self, "Detection Result", 
                                    "No heart shape detected. Try removing background or use a clearer heart image.")
                
        except Exception as e:
            print(f"Error detecting heart: {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            self.progress_bar.setVisible(False)
    
    def highlight_heart_contour(self):
        """Draw the detected heart contour on the image"""
        if self.heart_contour is None or self.heart_contour.shape[0] < 3:
            return
            
        # Create a copy of the current image
        image = self.processed_image.copy() if self.processed_image is not None else self.original_image.copy()
        
        # Draw contour on the image
        cv2.drawContours(image, [np.array(self.heart_contour, dtype=np.int32).reshape((-1, 1, 2))], 
                        0, (0, 255, 0, 255), 2)
        
        # Display the image with highlighted contour
        self.display_image(image)
    
    def convert_to_3d(self):
        """Convert to 3D heart based on selected model"""
        if self.heart_contour is None:
            # Try to detect heart first
            self.detect_heart()
            
            if self.heart_contour is None:
                QMessageBox.warning(self, "Error", "No heart shape detected")
                return
        
        # Get model type
        model_idx = self.model_selector.currentIndex()
        model_type = ["anatomical", "stylized", "extrusion", "rounded"][model_idx]
        
        # Get depth value
        depth = self.depth_slider.value() / 100.0
        
        # Set detail level based on checkbox
        if self.detail_checkbox.isChecked():
            self.converter.segments_horizontal = 64
            self.converter.segments_vertical = 48
        else:
            self.converter.segments_horizontal = 32
            self.converter.segments_vertical = 24
        
        # Start processing thread
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # Disable convert button during processing
        self.convert_button.setEnabled(False)
        
        # Create and start processing thread
        self.process_thread = ProcessThread(
            self.converter,
            self.heart_contour,
            depth,
            self.heart_color,
            model_type
        )
        
        # Connect signals
        self.process_thread.update_progress.connect(self.update_progress)
        self.process_thread.processing_done.connect(self.process_complete)
        
        # Start processing
        self.process_thread.start()
    
    def update_progress(self, value):
        """Update progress bar"""
        self.progress_bar.setValue(value)
    
    def process_complete(self, mesh):
        """Handle when processing is complete"""
        self.progress_bar.setVisible(False)
        self.convert_button.setEnabled(True)
        
        if mesh is not None:
            # Store mesh and display
            self.current_mesh = mesh
            self.display_mesh(mesh)
            self.export_button.setEnabled(True)
            
            QMessageBox.information(self, "Success", "3D heart model created successfully!")
        else:
            QMessageBox.warning(self, "Error", "Failed to create 3D heart model")
    
    def update_depth(self):
        """Update depth/volume label"""
        value = self.depth_slider.value() / 100.0
        self.depth_label.setText(f"Heart Size: {value:.2f}")
    
    def update_detail_level(self):
        """Update detail level based on checkbox"""
        pass  # Will apply on next conversion
    
    def update_shading(self):
        """Update shading mode"""
        if self.current_mesh is not None:
            self.display_mesh(self.current_mesh)
    
    def display_image(self, image):
        """Display an image in the UI"""
        height, width = image.shape[:2]
        
        if image.shape[2] == 4:  # BGRA
            fmt = QImage.Format.Format_RGBA8888
            bytes_per_line = 4 * width
            qimg = QImage(cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA).data, width, height, bytes_per_line, fmt)
        else:  # BGR
            fmt = QImage.Format.Format_RGB888
            bytes_per_line = 3 * width
            qimg = QImage(cv2.cvtColor(image, cv2.COLOR_BGR2RGB).data, width, height, bytes_per_line, fmt)
        
        pixmap = QPixmap.fromImage(qimg)
        self.image_label.setPixmap(pixmap.scaled(
            400, 400, Qt.AspectRatioMode.KeepAspectRatio
        ))
    
    def display_mesh(self, mesh):
        # Clear viewer
        self.viewer.clear()
        
        # Add grid
        grid = gl.GLGridItem()
        grid.setSize(2, 2)
        self.viewer.addItem(grid)
        
        # Extract mesh data
        vertices = mesh.vertices
        faces = mesh.faces
        
        # Center the mesh
        center = np.mean(vertices, axis=0)
        vertices = vertices - center
        
        # Scale to fit viewer
        scale = 2.0 / np.max(np.abs(vertices))
        vertices = vertices * scale
        
        # Get colors
        if hasattr(mesh.visual, 'vertex_colors'):
            colors = mesh.visual.vertex_colors
            if colors.max() > 1.0:
                colors = colors.astype(np.float32) / 255.0
        else:
            # Deep glossy red as default
            colors = np.ones((len(vertices), 4)) * [0.7, 0.05, 0.05, 1.0]
        
        # Create mesh item with appropriate shading
        mesh_item = gl.GLMeshItem(
            vertexes=vertices,
            faces=faces, 
            vertexColors=colors,
            smooth=self.smooth_checkbox.isChecked(),
            drawEdges=False,  # No edges for glossier appearance
            shader='shaded'
        )
        
        # Add to viewer
        self.viewer.addItem(mesh_item)
        
        # Adjust camera for better view of the heart
        self.viewer.setCameraPosition(distance=4, elevation=30, azimuth=45)
    
    def export_model(self):
        """Export 3D heart model to file"""
        if self.current_mesh is None:
            return
            
        file_name, _ = QFileDialog.getSaveFileName(
            self, "Save 3D Heart", "", "STL Files (*.stl);;OBJ Files (*.obj);;GLTF Files (*.gltf)"
        )
        
        if file_name:
            # Show progress
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(10)
            
            try:
                # Add extension if not specified
                if not file_name.endswith(('.stl', '.obj', '.gltf')):
                    file_name += '.stl'
                    
                # Export based on file type
                if file_name.endswith('.stl'):
                    self.current_mesh.export(file_name, file_type='stl')
                elif file_name.endswith('.obj'):
                    self.current_mesh.export(file_name, file_type='obj')
                elif file_name.endswith('.gltf'):
                    self.current_mesh.export(file_name, file_type='gltf')
                
                self.progress_bar.setValue(100)
                QMessageBox.information(self, "Success", f"Model saved to {file_name}")
                
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to export model: {str(e)}")
                
            finally:
                self.progress_bar.setVisible(False)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AdvancedHeartConverterApp()
    window.show()
    sys.exit(app.exec())