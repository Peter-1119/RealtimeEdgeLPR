from PyQt5 import QtWidgets, QtGui
import numpy  as np
import cv2

def qt_img(image, box_WH, mask = False, path = True):
    box_W, box_H = box_WH  # Get label box width & height (Optional)
    if mask:
        image = (np.ones((box_H, box_W, 3)) * 30).astype(np.uint8)
    elif path:
        image = cv2.imread(image)  # Read image
    
    image = cv2.resize(image, (box_W, box_H), interpolation=cv2.INTER_AREA)  # Zoom out the image
    height, width, channel = image.shape  # Get image width & channel & channel
    bytesPerline = 3 * width  # Channels per width

    qimg = QtGui.QImage(image, width, height, bytesPerline, QtGui.QImage.Format_RGB888).rgbSwapped()  # change img from cv2 format to qt format
    return QtGui.QPixmap.fromImage(qimg)  # output qt format image array

class QLabelComponent(QtWidgets.QLabel):
    def __init__(self, parent, geometry, text = None, align = None):
        super().__init__(parent)
        
        self.setGeometry(geometry['x'], geometry['y'], geometry['w'], geometry['h'])
        
        if text != None:
            self.setText(text)
            
        if align != None:
            self.setAlignment(align)

class QPushButtonComponent(QtWidgets.QPushButton):
    def __init__(self, parent, geometry, text = None, align = None):
        super().__init__(parent)
        
        self.setGeometry(geometry['x'], geometry['y'], geometry['w'], geometry['h'])
        
        if text != None:
            self.setText(text)
            
        if align != None:
            self.setAlignment(align)

class QLineEditComponent(QtWidgets.QLineEdit):
    def __init__(self, parent, geometry, align = None):
        super().__init__(parent)
        
        self.setGeometry(geometry['x'], geometry['y'], geometry['w'], geometry['h'])
        
        if align != None:
            self.setAlignment(align)
            
class QComboBoxComponent(QtWidgets.QComboBox):
    def __init__(self, parent, geometry, align = None):
        super().__init__(parent)
        
        self.setGeometry(geometry['x'], geometry['y'], geometry['w'], geometry['h'])
        
        if align != None:
            self.setAlignment(align)