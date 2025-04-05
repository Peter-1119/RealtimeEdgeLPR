from PyQt5 import QtCore, QtWidgets
from WorkWidgets.utils import *
import sys


class StreamWidget(QtWidgets.QWidget):
    streaming_sig = QtCore.pyqtSignal(dict)
    
    def __init__(self):
        super().__init__()
        
        self.initStreamUI()
        self.streaming_sig.connect(self.show_stream_and_capture)
        self.captures = []
        
    def initStreamUI(self):
        self.SwitchWindowBtn = QPushButtonComponent(self, {'x': 1590, 'y': 15, 'w': 220, 'h': 50}, "Search Window")
        self.SwitchWindowBtn.setStyleSheet('font-size: 14pt')
        
        self.streamWindow = QLabelComponent(self, {'x': 60, 'y': 320, 'w': 1430, 'h': 680})
        self.streamWindow.setStyleSheet('background-color: rgb(40, 40, 40)')
        
        self.captureLabelList = []
        for columnIndex, column in enumerate(range(60, 1810, 370)):
            captureLabel = QLabelComponent(self, {'x': column, 'y': 80, 'w': 320, 'h': 200})
            captureLabel.setStyleSheet('background-color: rgb(40, 40, 40)')
            self.captureLabelList.append(captureLabel)
            
        for rowIndex, row in enumerate(range(320, 1040, 240)):
            captureLabel = QLabelComponent(self, {'x': column, 'y': row, 'w': 320, 'h': 200})
            captureLabel.setStyleSheet('background-color: rgb(40, 40, 40)')
            self.captureLabelList.append(captureLabel)
            
    def show_stream_and_capture(self, event):
        self.streamWindow.setPixmap(qt_img(event['video'], [1430, 680], path = False))
        
        for capture in event['captures']:
            if len(self.captures) == 8:
                self.captures.pop(0)
            self.captures.append(capture)
        
        for index, capture in enumerate(self.captures[::-1]):
            self.captureLabelList[index].setPixmap(qt_img(capture, [320, 200], path = False))


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    
    mainWidget = StreamWidget()
    mainWidget.showMaximized()
    mainWidget.show()
    
    sys.exit(app.exec_())