"""
This is a wrapper for your 3D converter application
that ensures it opens in fullscreen mode.

Save this as 'oneup_3d_converter.py'
"""
import sys
import os
import importlib.util
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer

def main():
    try:
        # Create QApplication
        app = QApplication(sys.argv)
        
        # Find your main application file
        # Check if the original converter code exists
        if os.path.exists('main3.py'):
            # Load the module from main3.py
            spec = importlib.util.spec_from_file_location('converter_module', 'main3.py')
            converter_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(converter_module)
            
            # Create the MainWindow instance
            window = converter_module.MainWindow()
            
            # Show the window first to ensure proper initialization
            window.show()
            
            # Then maximize it after a brief delay
            QTimer.singleShot(100, window.showMaximized)
            
            # Start the application
            sys.exit(app.exec())
        elif os.path.exists('paste-2.txt'):
            # If the main3.py doesn't exist but paste-2.txt does,
            # convert paste-2.txt to a temporary Python file and run that
            with open('paste-2.txt', 'r') as f:
                code = f.read()
            
            # Create a temporary module
            import types
            converter_module = types.ModuleType('converter_module')
            exec(code, converter_module.__dict__)
            
            # Create the MainWindow instance
            window = converter_module.MainWindow()
            
            # Show the window first to ensure proper initialization
            window.show()
            
            # Then maximize it after a brief delay
            QTimer.singleShot(100, window.showMaximized)
            
            # Start the application
            sys.exit(app.exec())
        else:
            # If neither file exists, show an error message
            from PyQt6.QtWidgets import QMessageBox
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Icon.Critical)
            msg.setText("Error: Could not find 3D converter application file")
            msg.setInformativeText("Please make sure 'main3.py' or 'paste-2.txt' exists in the same directory.")
            msg.setWindowTitle("File Not Found")
            msg.exec()
            sys.exit(1)
            
    except Exception as e:
        # Show error dialog
        from PyQt6.QtWidgets import QMessageBox
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Critical)
        msg.setText("Error launching 3D converter")
        msg.setInformativeText(str(e))
        msg.setWindowTitle("Error")
        msg.exec()
        sys.exit(1)

if __name__ == "__main__":
    main()