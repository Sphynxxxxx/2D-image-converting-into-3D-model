import sys
import os
import subprocess
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                           QPushButton, QLabel, QFrame, QSizePolicy, QMessageBox)
from PyQt6.QtGui import QPixmap, QFont, QCursor, QIcon, QColor
from PyQt6.QtCore import Qt, QSize, QProcess, QTimer

class CardWidget(QFrame):
    def __init__(self, title, description, image_path, button_text, button_color="#3498db"):
        super().__init__()
        self.setObjectName("card")
        self.setStyleSheet("""
            #card {
                background-color: #2d2d2d;
                border-radius: 12px;
                min-height: 350px;
            }
        """)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        
        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 20)
        layout.setSpacing(15)
        
        # Image area
        self.image_label = QLabel()
        self.image_label.setFixedHeight(200)
        self.image_label.setStyleSheet("""
            QLabel {
                background-color: #1e1e1e;
                border-top-left-radius: 12px;
                border-top-right-radius: 12px;
            }
        """)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Try to load image, use placeholder if not found
        if os.path.exists(image_path):
            pixmap = QPixmap(image_path)
            self.image_label.setPixmap(pixmap.scaled(
                400, 200,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            ))
        else:
            # Use placeholder color if image not found
            self.image_label.setText("📷")
            self.image_label.setFont(QFont("Arial", 36))
        
        # Title
        title_label = QLabel(title)
        title_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("color: white;")
        
        # Description
        desc_label = QLabel(description)
        desc_label.setWordWrap(True)
        desc_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc_label.setStyleSheet("color: #95a5a6;")
        desc_label.setContentsMargins(20, 0, 20, 0)
        
        # Button
        self.button = QPushButton(button_text)
        self.button.setFixedHeight(40)
        self.button.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.button.setStyleSheet(f"""
            QPushButton {{
                background-color: {button_color};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {self._darken_color(button_color)};
            }}
            QPushButton:pressed {{
                background-color: {self._darken_color(button_color, factor=30)};
            }}
        """)
        
        # Add widgets to layout
        layout.addWidget(self.image_label)
        layout.addWidget(title_label)
        layout.addWidget(desc_label)
        
        # Button container for margin control
        button_container = QWidget()
        button_layout = QHBoxLayout(button_container)
        button_layout.setContentsMargins(20, 0, 20, 0)
        button_layout.addWidget(self.button)
        
        layout.addWidget(button_container)
    
    def _darken_color(self, hex_color, factor=20):
        """Darken a hex color by reducing RGB values by factor%"""
        hex_color = hex_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        factor = 1 - factor/100
        r = max(0, int(r * factor))
        g = max(0, int(g * factor))
        b = max(0, int(b * factor))
        return f"#{r:02x}{g:02x}{b:02x}"

class LandingPage(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OneUp: Converting 2D Images Into 3D Models")
        self.setGeometry(100, 100, 1100, 800)
        self.setMinimumSize(900, 700)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            QWidget {
                font-family: Arial, sans-serif;
                color: white;
            }
            QLabel#app-title {
                font-size: 28px;
                font-weight: bold;
                color: white;
            }
            QLabel#app-subtitle {
                font-size: 16px;
                color: #95a5a6;
            }
        """)
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        main_layout = QVBoxLayout(self.central_widget)
        main_layout.setSpacing(30)
        main_layout.setContentsMargins(40, 40, 40, 40)
        
        # Header with logo and title
        header_layout = QHBoxLayout()
        
        # Logo area
        logo_frame = QFrame()
        logo_frame.setFixedSize(90, 90)
        logo_frame.setStyleSheet("""
            background-color: black;
            border-radius: 10px;
        """)
        
        logo_layout = QVBoxLayout(logo_frame)
        logo_label = QLabel("OneUp")
        logo_label.setFont(QFont("Arial", 15, QFont.Weight.Bold))
        logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        logo_layout.addWidget(logo_label)
        
        # Title area
        title_layout = QVBoxLayout()
        title_layout.setSpacing(5)
        
        title_label = QLabel("OneUp")
        title_label.setObjectName("app-title")
        
        subtitle_label = QLabel("Converting 2D Images Into 3D Models")
        subtitle_label.setObjectName("app-subtitle")
        
        title_layout.addWidget(title_label)
        title_layout.addWidget(subtitle_label)
        
        header_layout.addWidget(logo_frame)
        header_layout.addSpacing(15)
        header_layout.addLayout(title_layout)
        header_layout.addStretch()
        
        # Hero section
        hero_label = QLabel("Transform 2D Images into 3D Models")
        hero_label.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        hero_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        description_label = QLabel("OneUp uses advanced digital image processing to convert your 2D images into high-quality 3D models ready for printing, design, or digital use.")
        description_label.setWordWrap(True)
        description_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        description_label.setStyleSheet("color: #95a5a6; font-size: 14px; max-width: 800px;")
        description_label.setContentsMargins(100, 0, 100, 0)
        
        # Cards container
        cards_layout = QHBoxLayout()
        cards_layout.setSpacing(30)
        
        # Shape converter card
        shape_card = CardWidget(
            "3D Shape Converter",
            "",
            "shape.png",  # Path to image - will show placeholder if not found
            "Open Shape Converter",
            "#3498db"
        )
        
        # Logo converter card
        logo_card = CardWidget(
            "3D Logo Converter",
            "",
            "logo.png",  # Path to image - will show placeholder if not found
            "Open Logo Converter",
            "#2ecc71"
        )
        
        # Connect buttons to actions
        shape_card.button.clicked.connect(self.open_shape_converter)
        logo_card.button.clicked.connect(self.open_logo_converter)
        
        cards_layout.addWidget(shape_card)
        cards_layout.addWidget(logo_card)
        
        
        # Footer
        footer_label = QLabel("© 2025 OneUp: Converting 2D Images Into 3D Model. All rights reserved.")
        footer_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        footer_label.setStyleSheet("color: #95a5a6; margin-top: 20px;")
        
        # Add all sections to main layout
        main_layout.addLayout(header_layout)
        main_layout.addSpacing(20)
        main_layout.addWidget(hero_label)
        main_layout.addWidget(description_label)
        main_layout.addSpacing(20)
        main_layout.addLayout(cards_layout)
        main_layout.addSpacing(40)
        main_layout.addSpacing(20)
        main_layout.addStretch()
        main_layout.addWidget(footer_label)
    
    def create_feature_widget(self, icon, title, description):
        feature = QFrame()
        feature.setStyleSheet("""
            QFrame {
                background-color: #2d2d2d;
                border-radius: 10px;
                padding: 15px;
            }
        """)
        
        layout = QVBoxLayout(feature)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        icon_label = QLabel(icon)
        icon_label.setFont(QFont("Arial", 24))
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        title_label = QLabel(title)
        title_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        desc_label = QLabel(description)
        desc_label.setWordWrap(True)
        desc_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc_label.setStyleSheet("color: #95a5a6;")
        
        layout.addWidget(icon_label)
        layout.addWidget(title_label)
        layout.addWidget(desc_label)
        
        return feature
    
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
        
    def open_logo_converter(self):
        """Open the logo converter application and close the landing page"""
        print("Opening 3D Logo Converter...")
        
        try:
            converter_script = "main3.py"  # Direct path to your 3D converter script
            
            # Check if the file exists
            if os.path.exists(converter_script):
                # Create loading message
                msg = QLabel("Launching 3D Logo Converter...")
                msg.setStyleSheet("background-color: #2ecc71; color: white; padding: 10px; border-radius: 5px;")
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
                
                # Start the logo converter directly using subprocess
                # This method works better for launching fullscreen applications
                if sys.platform == 'win32':  # For Windows
                    subprocess.Popen([sys.executable, converter_script], 
                                   creationflags=subprocess.CREATE_NEW_CONSOLE)
                else:  # For macOS and Linux
                    subprocess.Popen([sys.executable, converter_script])
                
                # Close the landing page after a short delay
                QTimer.singleShot(1000, self.close)
                
            else:
                # If file doesn't exist, show error message
                msg = QLabel("Error: main3.py not found!")
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
                QApplication.processEvents()
                import time
                time.sleep(3)
                msg.hide()
        except Exception as e:
            print(f"Error opening logo converter: {str(e)}")
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


def main():
    app = QApplication(sys.argv)
    window = LandingPage()
    
    # Use a timer to maximize the window after it's shown
    window.show()
    QTimer.singleShot(500, window.showMaximized)
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()