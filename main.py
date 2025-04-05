from PyQt5 import QtWidgets
from WorkWidgets.MainWidget import MainWidget
from models.LPR import LPR
import sys

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    
    # Declare model and detect class
    save_classes_no_normal_car = [0, 1, 2, 5, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
    attach_index = [2, 8]
    lpr = LPR("models/weights/yolo/best.engine", "models/classes.txt", "models/weights/crnn/BestNetCN.pth", attach_index, save_classes_no_normal_car)
    
    # Create main window
    mainWidget = MainWidget(lpr)
    mainWidget.showMaximized()
    mainWidget.show()
    
    sys.exit(app.exec_())