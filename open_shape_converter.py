def open_shape_converter(self):
    """Open the 2D to 3D shape converter application and close the landing page"""
    print("Opening 2D to 3D Shape Converter...")
    
    try:
        converter_script = "main.py"  # Path to your shape converter script
        
        # Check if the file exists
        if os.path.exists(converter_script):
            # Create loading message
            msg = QLabel("Launching 2D to 3D Shape Converter...")
            msg.setStyleSheet("background-color: #3498db; color: white; padding: 10px; border-radius: 5px;")
            msg.setFixedSize(300, 40)
            msg.setAlignment(Qt.AlignmentFlag.AlignCenter)
            
            # Position at the center of the window
            msg.setParent(self.central_widget)
            msg.move(
                (self.central_widget.width() - msg.width()) // 2,
                (self.central_widget.height() - msg.height()) // 2
            )
            msg.show()
            QApplication.processEvents()
            
            # Start the shape converter directly using subprocess
            if sys.platform == 'win32':  # For Windows
                subprocess.Popen([sys.executable, converter_script], 
                               creationflags=subprocess.CREATE_NEW_CONSOLE)
            else:  # For macOS and Linux
                subprocess.Popen([sys.executable, converter_script])
            
            # Close the landing page after a short delay
            QTimer.singleShot(1000, self.close)
            
        else:
            # If file doesn't exist, show error message
            msg = QLabel("Error: main.py not found!")
            msg.setStyleSheet("background-color: #e74c3c; color: white; padding: 10px; border-radius: 5px;")
            msg.setFixedSize(350, 40)
            msg.setAlignment(Qt.AlignmentFlag.AlignCenter)
            
            # Position at the center of the window
            msg.setParent(self.central_widget)
            msg.move(
                (self.central_widget.width() - msg.width()) // 2,
                (self.central_widget.height() - msg.height()) // 2
            )
            msg.show()
            
            # Auto-hide after 3 seconds
            QTimer.singleShot(3000, msg.hide)
    except Exception as e:
        print(f"Error opening shape converter: {str(e)}")
        # Show error message
        msg = QLabel(f"Error: {str(e)}")
        msg.setStyleSheet("background-color: #e74c3c; color: white; padding: 10px; border-radius: 5px;")
        msg.setFixedSize(350, 40)
        msg.setAlignment(Qt.AlignmentFlag.AlignCenter)
        msg.setParent(self.central_widget)
        msg.move(
            (self.central_widget.width() - msg.width()) // 2,
            (self.central_widget.height() - msg.height()) // 2
        )
        msg.show()
        QTimer.singleShot(3000, msg.hide)