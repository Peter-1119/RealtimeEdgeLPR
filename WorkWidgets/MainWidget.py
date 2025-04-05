from PyQt5 import QtCore, QtWidgets
from WorkWidgets.SearchWidgetUI import SearchWidget
from WorkWidgets.StreamWidgetUI import StreamWidget
import sys

class MainWidget(QtWidgets.QWidget):
    def __init__(self, model):
        super().__init__()
        
        self.widgets = {
            "stream": {'widget': StreamWidget(), 'switch_window': 1},
            "search": {'widget': SearchWidget(), 'switch_window': 0},
        }
        self.widgets['stream']['widget'].SwitchWindowBtn.clicked.connect(lambda: self.switch_window(currentWindow = "stream"))
        self.widgets['search']['widget'].SwitchWindowBtn.clicked.connect(lambda: self.switch_window(currentWindow = "search"))

        self.stackLayout = QtWidgets.QStackedLayout()
        self.stackLayout.addWidget(self.widgets['stream']['widget'])
        self.stackLayout.addWidget(self.widgets['search']['widget'])
        
        self.setLayout(self.stackLayout)
        
        # Start detect model
        self.model = model
        self.model.raise_signal = self.widgets['stream']['widget'].streaming_sig
        self.model.start()
        
    def switch_window(self, currentWindow):
        self.stackLayout.setCurrentIndex(self.widgets[currentWindow]['switch_window'])

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)

    mainWidget = MainWidget()
    mainWidget.showMaximized()
    mainWidget.show()
    
    sys.exit(app.exec_())