import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from PySide6.QtWidgets import QApplication
from gui.main_window import MainWindow

def capture():
    app = QApplication(sys.argv)
    window = MainWindow(stream_url="http://10.3.3.193:8000/stream")
    window.show()
    # Let it draw
    QApplication.processEvents()
    
    # Grab the screenshot
    pixmap = window.grab()
    pixmap.save("phase0_screenshot.png")
    app.quit()

if __name__ == "__main__":
    capture()
