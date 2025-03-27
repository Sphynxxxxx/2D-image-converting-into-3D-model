
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import os
import threading
from PIL import Image, ImageTk
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from shape_to_3d import ShapeTo3D  # Import the existing class

class ShapeTo3DApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Shape to 3D Converter")
        self.root.geometry("1200x800")
        self.root.configure(bg="#f0f0f0")
        
        # Initialize the converter
        self.converter = ShapeTo3D()
        
        # Set up variables
        self.image_path = None
        self.threshold = tk.IntVar(value=127)
        self.blur_size = tk.IntVar(value=5)
        self.point_density = tk.DoubleVar(value=0.5)
        self.point_size = tk.IntVar(value=5)
        self.elevation = tk.IntVar(value=30)
        self.azimuth = tk.IntVar(value=30)
        self.conversion_mode = tk.StringVar(value="General 3D")
        self.resolution = tk.IntVar(value=50)
        
        # Create UI layout
        self.create_ui()
    
    def create_ui(self):
        # Create main frames
        left_frame = tk.Frame(self.root, bg="#f0f0f0", padx=10, pady=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)
        
        right_frame = tk.Frame(self.root, bg="#f0f0f0", padx=10, pady=10)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Left frame - Controls and 2D preview
        control_frame = tk.LabelFrame(left_frame, text="Controls", bg="#f0f0f0", padx=10, pady=10)
        control_frame.pack(fill=tk.X, pady=5)
        
        preview_frame = tk.LabelFrame(left_frame, text="2D Preview", bg="#f0f0f0", padx=10, pady=10)
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Right frame - 3D visualization
        visualization_frame = tk.LabelFrame(right_frame, text="3D Visualization", bg="#f0f0f0", padx=10, pady=10)
        visualization_frame.pack(fill=tk.BOTH, expand=True)
        
        # Load image button
        load_button = ttk.Button(control_frame, text="Load Image", command=self.load_image)
        load_button.grid(row=0, column=0, columnspan=2, pady=5, sticky="ew")
        
        # Create sample shape buttons
        samples_frame = tk.LabelFrame(control_frame, text="Create Sample", bg="#f0f0f0", padx=5, pady=5)
        samples_frame.grid(row=1, column=0, columnspan=2, pady=5, sticky="ew")
        
        circle_button = ttk.Button(samples_frame, text="Circle", command=lambda: self.create_sample("circle"))
        circle_button.grid(row=0, column=0, padx=5, pady=5)
        
        square_button = ttk.Button(samples_frame, text="Square", command=lambda: self.create_sample("square"))
        square_button.grid(row=0, column=1, padx=5, pady=5)
        
        triangle_button = ttk.Button(samples_frame, text="Triangle", command=lambda: self.create_sample("triangle"))
        triangle_button.grid(row=0, column=2, padx=5, pady=5)
        
        # Conversion mode selector
        mode_label = ttk.Label(control_frame, text="Conversion Mode:")
        mode_label.grid(row=2, column=0, sticky="w", pady=5)
        
        mode_combo = ttk.Combobox(control_frame, textvariable=self.conversion_mode, values=["General 3D", "Circle to Sphere"])
        mode_combo.grid(row=2, column=1, sticky="ew", pady=5)
        mode_combo.bind("<<ComboboxSelected>>", self.update_parameter_visibility)
        
        # Parameters
        params_frame = tk.LabelFrame(control_frame, text="Parameters", bg="#f0f0f0", padx=5, pady=5)
        params_frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=5)
        
        # Threshold
        threshold_label = ttk.Label(params_frame, text="Threshold:")
        threshold_label.grid(row=0, column=0, sticky="w", pady=2)
        threshold_slider = ttk.Scale(params_frame, from_=0, to=255, variable=self.threshold, orient=tk.HORIZONTAL)
        threshold_slider.grid(row=0, column=1, sticky="ew", pady=2)
        threshold_value = ttk.Label(params_frame, textvariable=self.threshold)
        threshold_value.grid(row=0, column=2, padx=5)
        
        # Blur Size
        blur_label = ttk.Label(params_frame, text="Blur Size:")
        blur_label.grid(row=1, column=0, sticky="w", pady=2)
        blur_slider = ttk.Scale(params_frame, from_=1, to=21, variable=self.blur_size, orient=tk.HORIZONTAL)
        blur_slider.grid(row=1, column=1, sticky="ew", pady=2)
        blur_value = ttk.Label(params_frame, textvariable=self.blur_size)
        blur_value.grid(row=1, column=2, padx=5)
        
        # Point Density
        density_label = ttk.Label(params_frame, text="Point Density:")
        density_label.grid(row=2, column=0, sticky="w", pady=2)
        density_slider = ttk.Scale(params_frame, from_=0.1, to=1.0, variable=self.point_density, orient=tk.HORIZONTAL)
        density_slider.grid(row=2, column=1, sticky="ew", pady=2)
        density_value = ttk.Label(params_frame, textvariable=self.point_density)
        density_value.grid(row=2, column=2, padx=5)
        
        # Point Size
        point_size_label = ttk.Label(params_frame, text="Point Size:")
        point_size_label.grid(row=3, column=0, sticky="w", pady=2)
        point_size_slider = ttk.Scale(params_frame, from_=1, to=20, variable=self.point_size, orient=tk.HORIZONTAL)
        point_size_slider.grid(row=3, column=1, sticky="ew", pady=2)
        point_size_value = ttk.Label(params_frame, textvariable=self.point_size)
        point_size_value.grid(row=3, column=2, padx=5)
        
        # Resolution (for sphere)
        self.resolution_label = ttk.Label(params_frame, text="Resolution:")
        self.resolution_label.grid(row=4, column=0, sticky="w", pady=2)
        self.resolution_slider = ttk.Scale(params_frame, from_=10, to=100, variable=self.resolution, orient=tk.HORIZONTAL)
        self.resolution_slider.grid(row=4, column=1, sticky="ew", pady=2)
        self.resolution_value = ttk.Label(params_frame, textvariable=self.resolution)
        self.resolution_value.grid(row=4, column=2, padx=5)
        
        # View controls
        view_frame = tk.LabelFrame(control_frame, text="View Controls", bg="#f0f0f0", padx=5, pady=5)
        view_frame.grid(row=4, column=0, columnspan=2, sticky="ew", pady=5)
        
        # Elevation
        elev_label = ttk.Label(view_frame, text="Elevation:")
        elev_label.grid(row=0, column=0, sticky="w", pady=2)
        elev_slider = ttk.Scale(view_frame, from_=0, to=90, variable=self.elevation, orient=tk.HORIZONTAL)
        elev_slider.grid(row=0, column=1, sticky="ew", pady=2)
        elev_value = ttk.Label(view_frame, textvariable=self.elevation)
        elev_value.grid(row=0, column=2, padx=5)
        
        # Azimuth
        azim_label = ttk.Label(view_frame, text="Azimuth:")
        azim_label.grid(row=1, column=0, sticky="w", pady=2)
        azim_slider = ttk.Scale(view_frame, from_=0, to=360, variable=self.azimuth, orient=tk.HORIZONTAL)
        azim_slider.grid(row=1, column=1, sticky="ew", pady=2)
        azim_value = ttk.Label(view_frame, textvariable=self.azimuth)
        azim_value.grid(row=1, column=2, padx=5)
        
        # Process button
        process_button = ttk.Button(control_frame, text="Convert to 3D", command=self.convert_to_3d)
        process_button.grid(row=5, column=0, columnspan=2, pady=10, sticky="ew")
        
        # Save button
        save_button = ttk.Button(control_frame, text="Save 3D Image", command=self.save_3d_image)
        save_button.grid(row=6, column=0, columnspan=2, pady=5, sticky="ew")
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Image preview canvas
        self.preview_canvas = tk.Canvas(preview_frame, bg="white")
        self.preview_canvas.pack(fill=tk.BOTH, expand=True)
        
        # 3D visualization placeholder
        self.visualization_placeholder = tk.Frame(visualization_frame, bg="white")
        self.visualization_placeholder.pack(fill=tk.BOTH, expand=True)
        
        # Initialize UI state
        self.update_parameter_visibility()
        
        # Configure grid weights for responsive layout
        for i in range(7):
            control_frame.grid_rowconfigure(i, weight=1)
        control_frame.grid_columnconfigure(1, weight=1)
    
    def update_parameter_visibility(self, event=None):
        mode = self.conversion_mode.get()
        if mode == "Circle to Sphere":
            # Show resolution, hide point density and point size
            self.resolution_label.grid()
            self.resolution_slider.grid()
            self.resolution_value.grid()
        else:
            # Hide resolution, show point density and point size
            self.resolution_label.grid_remove()
            self.resolution_slider.grid_remove()
            self.resolution_value.grid_remove()
    
    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.image_path = file_path
            self.display_preview(file_path)
            self.status_var.set(f"Loaded image: {os.path.basename(file_path)}")
    
    def create_sample(self, shape_type):
        # Create a temporary directory if it doesn't exist
        temp_dir = os.path.join(os.path.expanduser("~"), ".shape_to_3d_temp")
        os.makedirs(temp_dir, exist_ok=True)
        
        size = 500
        file_path = os.path.join(temp_dir, f"{shape_type}.png")
        
        # Create black image
        img = np.zeros((size, size), np.uint8)
        
        if shape_type == "circle":
            # Draw circle
            center = (size // 2, size // 2)
            radius = 200
            cv2.circle(img, center, radius, 255, -1)
            
        elif shape_type == "square":
            # Draw square
            square_size = 300
            start_x = (size - square_size) // 2
            start_y = (size - square_size) // 2
            end_x = start_x + square_size
            end_y = start_y + square_size
            cv2.rectangle(img, (start_x, start_y), (end_x, end_y), 255, -1)
            
        elif shape_type == "triangle":
            # Draw triangle
            vertices = np.array([
                [size // 2, size // 4],
                [size // 4, 3 * size // 4],
                [3 * size // 4, 3 * size // 4]
            ], np.int32)
            cv2.fillPoly(img, [vertices], 255)
        
        cv2.imwrite(file_path, img)
        self.image_path = file_path
        self.display_preview(file_path)
        
        # If it's a circle, automatically set to circle mode
        if shape_type == "circle":
            self.conversion_mode.set("Circle to Sphere")
        else:
            self.conversion_mode.set("General 3D")
            
        self.update_parameter_visibility()
        self.status_var.set(f"Created {shape_type} sample")
    
    def display_preview(self, image_path):
        try:
            # Open image with PIL (supports more formats than PhotoImage)
            pil_img = Image.open(image_path)
            
            # Resize image to fit the preview canvas while maintaining aspect ratio
            canvas_width = self.preview_canvas.winfo_width() or 300
            canvas_height = self.preview_canvas.winfo_height() or 300
            
            # Calculate new dimensions
            width, height = pil_img.size
            ratio = min(canvas_width/width, canvas_height/height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            
            # Resize image
            pil_img = pil_img.resize((new_width, new_height), Image.LANCZOS)
            
            # Convert to PhotoImage
            self.preview_image = ImageTk.PhotoImage(pil_img)
            
            # Clear previous image and display new one
            self.preview_canvas.delete("all")
            self.preview_canvas.config(width=new_width, height=new_height)
            self.preview_canvas.create_image(new_width//2, new_height//2, image=self.preview_image)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
            self.status_var.set("Error loading image")
    
    def convert_to_3d(self):
        if not self.image_path:
            messagebox.showwarning("No Image", "Please load an image first")
            return
            
        self.status_var.set("Converting to 3D...")
        
        # Start conversion in a separate thread to avoid UI freezing
        thread = threading.Thread(target=self._process_conversion)
        thread.daemon = True
        thread.start()
    
    def _process_conversion(self):
        try:
            # Clear previous visualization
            for widget in self.visualization_placeholder.winfo_children():
                widget.destroy()
                
            # Get parameters
            threshold = self.threshold.get()
            blur_size = self.blur_size.get()
            if blur_size % 2 == 0:  # Ensure blur size is odd
                blur_size += 1
            point_density = self.point_density.get()
            point_size = self.point_size.get()
            elevation = self.elevation.get()
            azimuth = self.azimuth.get()
            resolution = self.resolution.get()
            
            # Create figure for embedding
            fig = plt.figure(figsize=(8, 6))
            
            # Process based on selected mode
            if self.conversion_mode.get() == "Circle to Sphere":
                # Create sphere from circle
                fig, ax = self.converter.create_sphere_from_circle(
                    self.image_path, 
                    threshold=threshold,
                    blur_size=blur_size,
                    resolution=resolution
                )
                
                # Configure view
                ax.view_init(elev=elevation, azim=azimuth)
                
            else:
                # General 3D processing
                fig, ax = self.converter.process_shape(
                    self.image_path,
                    threshold=threshold,
                    blur_size=blur_size,
                    point_density=point_density,
                    point_size=point_size,
                    elevation=elevation,
                    azimuth=azimuth
                )
            
            # Embed figure in tkinter
            canvas = FigureCanvasTkAgg(fig, master=self.visualization_placeholder)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Store canvas for later use
            self.fig = fig
            self.canvas = canvas
            
            self.status_var.set("3D conversion complete")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to convert to 3D: {str(e)}")
            self.status_var.set(f"Error: {str(e)}")
    
    def save_3d_image(self):
        if not hasattr(self, 'fig'):
            messagebox.showwarning("No 3D Image", "Please convert to 3D first")
            return
            
        file_path = filedialog.asksaveasfilename(
            title="Save 3D Image",
            defaultextension=".png",
            filetypes=[
                ("PNG Image", "*.png"),
                ("JPEG Image", "*.jpg"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                self.fig.savefig(file_path, dpi=300, bbox_inches='tight')
                self.status_var.set(f"Saved 3D image to: {os.path.basename(file_path)}")
                messagebox.showinfo("Success", f"3D image saved to: {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save image: {str(e)}")
                self.status_var.set("Error saving image")

