import sys
import cv2
import numpy as np
import trimesh
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QFileDialog, QPushButton, QHBoxLayout
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import torch
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import open3d as o3d

# Load MiDaS model for monocular depth estimation
def load_midas_model():
    """
    Load the MiDaS model for depth estimation.
    """
    model_type = "DPT_Large"  # MiDaS v3 - Large (highest accuracy, slowest inference speed)
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    midas.eval()

    # Use GPU if available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas.to(device)

    # Load transforms for preprocessing
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.dpt_transform if model_type == "DPT_Large" else midas_transforms.small_transform

    return midas, transform, device

def estimate_depth(image, midas, transform, device):
    """
    Estimate depth from an image using the MiDaS model.
    """
    # Convert PIL Image to a tensor and apply the transform
    input_image = transform(image).to(device)

    # Perform inference
    with torch.no_grad():
        prediction = midas(input_image)

    # Resize the depth map to the original image size
    depth_map = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=image.size[::-1],  # (width, height)
        mode="bicubic",
        align_corners=False,
    ).squeeze().cpu().numpy()

    return depth_map

def create_3d_mesh(depth_map, scale=10):
    """
    Create a 3D mesh from the depth map.
    """
    height, width = depth_map.shape

    # Create a grid of vertices
    x = np.linspace(0, width, width)
    y = np.linspace(0, height, height)
    x, y = np.meshgrid(x, y)

    # Flatten the grid and add depth as the z-coordinate
    vertices = np.vstack((x.flatten(), y.flatten(), depth_map.flatten() * scale)).T

    # Create faces for the mesh
    faces = []
    for i in range(height - 1):
        for j in range(width - 1):
            # Define the indices of the four vertices of the current quad
            v1 = i * width + j
            v2 = v1 + 1
            v3 = v2 + width
            v4 = v1 + width

            # Create two triangles for the quad
            faces.append([v1, v2, v3])
            faces.append([v3, v4, v1])

    faces = np.array(faces)

    # Create the mesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    return mesh

def export_mesh(mesh, output_path):
    """
    Export the 3D mesh as an STL file.
    """
    mesh.export(output_path, file_type='stl')

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Initialize the MiDaS model
        self.midas, self.transform, self.device = load_midas_model()

        # Set up the GUI
        self.setWindowTitle("2D Image to 3D Sculpture")
        self.setGeometry(100, 100, 1200, 800)

        # Create a central widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Add a label for drag-and-drop instructions
        self.label = QLabel("Drag and drop an image here or click to open a file", self)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("font-size: 20px; color: #333;")
        self.layout.addWidget(self.label)

        # Add a button to open a file dialog
        self.open_button = QPushButton("Open Image", self)
        self.open_button.clicked.connect(self.open_file_dialog)
        self.layout.addWidget(self.open_button)

        # Add a horizontal layout for displaying the depth map and 3D preview
        self.display_layout = QHBoxLayout()
        self.layout.addLayout(self.display_layout)

        # Add a label to display the depth map
        self.depth_map_label = QLabel(self)
        self.depth_map_label.setAlignment(Qt.AlignCenter)
        self.display_layout.addWidget(self.depth_map_label)

        # Add a matplotlib canvas for 3D preview
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.display_layout.addWidget(self.canvas)

        # Enable drag-and-drop
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        """
        Handle drag enter events.
        """
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        """
        Handle drop events.
        """
        # Get the file path from the dropped URL
        file_path = event.mimeData().urls()[0].toLocalFile()

        # Process the image
        self.process_image(file_path)

    def process_image(self, file_path):
        """
        Process the image and display the result.
        """
        # Load the image
        image = Image.open(file_path).convert("RGB")

        # Estimate depth using MiDaS
        depth_map = estimate_depth(image, self.midas, self.transform, self.device)

        # Normalize the depth map for visualization
        depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # Display the depth map
        depth_image = QImage(depth_map_normalized.data, depth_map_normalized.shape[1], depth_map_normalized.shape[0], QImage.Format_Grayscale8)
        self.depth_map_label.setPixmap(QPixmap.fromImage(depth_image))

        # Create a 3D mesh
        mesh = create_3d_mesh(depth_map, scale=20)

        # Export the mesh as an STL file
        output_stl_path = "output_model.stl"
        export_mesh(mesh, output_stl_path)

        # Display the 3D mesh
        self.display_3d_mesh(mesh)

        print(f"3D model saved to {output_stl_path}")

    def display_3d_mesh(self, mesh):
        """
        Display the 3D mesh using Open3D.
        """
        # Convert Trimesh to Open3D
        vertices = np.array(mesh.vertices)
        faces = np.array(mesh.faces)
        mesh_o3d = o3d.geometry.TriangleMesh()
        mesh_o3d.vertices = o3d.utility.Vector3dVector(vertices)
        mesh_o3d.triangles = o3d.utility.Vector3iVector(faces)

        # Visualize the mesh
        o3d.visualization.draw_geometries([mesh_o3d])

    def open_file_dialog(self):
        """
        Open a file dialog to select an image.
        """
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            self.process_image(file_path)

# Run the application
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())