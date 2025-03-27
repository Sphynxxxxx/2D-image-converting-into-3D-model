"""
This script launches the OneUp 3D Logo Converter application.
Save this as 'logo_converter.py' in the same directory as your landing page.
"""
import sys
import os
import subprocess

def main():
    try:
        # Define the path to the main application file
        main_file = 'main3.py'
        
        # Check if the file exists
        if os.path.exists(main_file):
            # Launch the application as a separate process
            if sys.platform == 'win32':  # Windows
                subprocess.Popen([sys.executable, main_file], 
                               creationflags=subprocess.CREATE_NEW_CONSOLE)
            else:  # Mac/Linux
                subprocess.Popen([sys.executable, main_file])
                
            # Signal success to the parent process (landing page)
            print("SUCCESS")  # This will be captured by the parent process
        else:
            print("ERROR: Main application file not found")
            
    except Exception as e:
        print(f"ERROR: {str(e)}")

if __name__ == "__main__":
    main()