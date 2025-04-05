from PyQt5 import QtCore, QtGui, QtWidgets
from WorkWidgets.utils import *
import sys, os


class SearchWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        
        self.page = 0
        self.vehicles = []
        
        self.initSearchUI()
        self.initInfoUI()
        self.setStyleSheet("QLabel{font-size: 11pt;}")
        
    def initSearchUI(self):
        self.SwitchWindowBtn = QPushButtonComponent(self, {'x': 1550, 'y': 100, 'w': 220, 'h': 50}, "Stream Window")
        self.SwitchWindowBtn.setStyleSheet("font-size: 14pt")
        
        SearchBackground = QLabelComponent(self, {'x': 450, 'y': 50, 'w': 1020, 'h': 150})
        SearchBackground.setFrameShape(QtWidgets.QFrame.Box)
        SearchBackground.setStyleSheet("background-color: rgb(253, 255, 206)")
        
        carPlateLabel = QLabelComponent(self, {'x': 500, 'y': 110, 'w': 100, 'h': 30}, "Input Car Plate: ")
        self.carPlateLabel = QLineEditComponent(self, {'x': 610, 'y': 110, 'w': 200, 'h': 30}, QtCore.Qt.AlignCenter)
        
        carCategory = QLabelComponent(self, {'x': 820, 'y': 60, 'w': 100, 'h': 30}, "Car Category: ", QtCore.Qt.AlignCenter)
        self.carCategoryComboBox = QComboBoxComponent(self, {'x': 920, 'y': 60, 'w': 100, 'h': 30})
        self.carCategoryComboBox.addItems(["All", 'motor', 'sportcar', 'car', 'sedan', 'suv', 'mpv', 'taxi', 'van', 'minitruck', 'mediumtruck', 'heavytruck', 'graveltruck'])
        carColor= QLabelComponent(self, {'x': 820, 'y': 110, 'w': 100, 'h': 30}, "Car Color: ", QtCore.Qt.AlignCenter)
        self.carColorComboBox = QComboBoxComponent(self, {'x': 920, 'y': 110, 'w': 100, 'h': 30})
        self.carColorComboBox.addItems(['All', 'silver', 'white', 'black', 'yellow', 'blue'])
        carBrand = QLabelComponent(self, {'x': 820, 'y': 160, 'w': 100, 'h': 30}, "Car Brand: ", QtCore.Qt.AlignCenter)
        self.carBrandComboBox = QComboBoxComponent(self, {'x': 920, 'y': 160, 'w': 100, 'h': 30})
        self.carBrandComboBox.addItems(["All", "toyota", 'ford', 'honda', 'BMW'])
        
        Place = QLabelComponent(self, {'x': 1040, 'y': 110, 'w': 90, 'h': 30}, "Search Place: ")
        self.country = QComboBoxComponent(self, {'x': 1140, 'y': 110, 'w': 80, 'h': 30})
        self.country.addItems(["All", "Taipei", "New Taipei", "Yilan", "Taoyuan", "Hsinchu", "Miaoli", "Taichung", "Changhua", "Nantou", "Yunlin", "Chiayi", "Tainan", "Kaohsiung", "Pingtung", "Taitung", "Hualien"])
        self.city = QComboBoxComponent(self, {'x': 1230, 'y': 110, 'w': 80, 'h': 30})
        self.city.addItems(["All"])
        
        self.searchBtn = QPushButtonComponent(self, {'x': 1360, 'y': 110, 'w': 80, 'h': 30}, "Search")
        self.searchBtn.clicked.connect(self.searchVehicle)
        
    def initInfoUI(self):
        self.frontPage = QPushButtonComponent(self, {'x': 20, 'y': 560, 'w': 60, 'h': 90})
        self.frontPage.setIcon(QtGui.QIcon("WorkWidgets/Resources/left-arrow.png"))
        self.frontPage.setIconSize(QtCore.QSize(60,90))
        self.frontPage.clicked.connect(self.turn_front_page)
        self.nextPage = QPushButtonComponent(self, {'x': 1840, 'y': 560, 'w': 60, 'h': 90})
        self.nextPage.setIcon(QtGui.QIcon("WorkWidgets/Resources/right-arrow.png"))
        self.nextPage.setIconSize(QtCore.QSize(60,90))
        self.nextPage.clicked.connect(self.turn_next_page)
        
        infoBackgroundColor = ["rgb(255, 255, 255)", "rgb(147, 198, 230)", "rgb(250, 219, 216)"]
        self.InfoLabelList = []
        for rowIndex, row in enumerate(range(250, 1000, 250)):
            for columnIndex, column in enumerate(range(100, 1850, 350)):
                InfoBackground = QLabelComponent(self, {'x': column, 'y': row, 'w': 320, 'h': 210})
                InfoBackground.setStyleSheet(f"background-color: {infoBackgroundColor[rowIndex]}")
                
                photoLabel = QLabelComponent(self, {'x': column + 10, 'y': row + 15, 'w': 130, 'h': 180})
                photoLabel.setStyleSheet("background-color: rgb(30, 30, 30)")
                
                Carplate = QLabelComponent(self, {'x': column + 145, 'y': row + 5, 'w': 170, 'h': 40}, "Carplate:")
                Category = QLabelComponent(self, {'x': column + 145, 'y': row + 45, 'w': 170, 'h': 40}, "Category:")
                Datetime = QLabelComponent(self, {'x': column + 145, 'y': row + 85, 'w': 170, 'h': 40}, "Datetime:")
                Position = QLabelComponent(self, {'x': column + 145, 'y': row + 125, 'w': 170, 'h': 40}, "Position:")
                Equipment = QLabelComponent(self, {'x': column + 145, 'y': row + 165, 'w': 170, 'h': 40}, "Equipment:")
                
                self.InfoLabelList.append({"photo": photoLabel, "CarPlate": Carplate, "Category": Category, "Datetime": Datetime, "Position": Position, "Equipment": Equipment})
    
    def showCapture(self, page):
        imgsCount = len(self.vehicles)
        
        for labelIndex, photoindex in enumerate(range(page * 15, page * 15 + 15)):
            if (photoindex >= imgsCount):
                self.InfoLabelList[labelIndex]['photo'].setPixmap(qt_img("", [130, 180], mask = True))
                self.InfoLabelList[labelIndex]['CarPlate'].setText(f"Carplate:")
                self.InfoLabelList[labelIndex]['Category'].setText(f"Category:")
                self.InfoLabelList[labelIndex]['Datetime'].setText(f"Datetime:")
                self.InfoLabelList[labelIndex]['Position'].setText("Position:")
                self.InfoLabelList[labelIndex]['Equipment'].setText("Equipment:")
            
            else:
                # print(f"({len(self.vehicles)})index: {index}")
                root, folder, filename = self.vehicles[photoindex].split("/")
                vehicleInfo = filename[:-4].split("_")
                
                track_id = vehicleInfo[0]
                vehicleTime = vehicleInfo[1]
                category = vehicleInfo[2]
                plateNumber = vehicleInfo[3]
                
                self.InfoLabelList[labelIndex]['photo'].setPixmap(qt_img(self.vehicles[photoindex], [130, 180]))
                self.InfoLabelList[labelIndex]['CarPlate'].setText(f"Carplate: {plateNumber}")
                self.InfoLabelList[labelIndex]['Category'].setText(f"Category: {category}")
                self.InfoLabelList[labelIndex]['Datetime'].setText(f"Datetime: {vehicleTime}")
                self.InfoLabelList[labelIndex]['Position'].setText("Position: New Taipei")
                self.InfoLabelList[labelIndex]['Equipment'].setText("Equipment: Device_1")
        
    def readRecord(self):
        vehicles = [f"Record/Motors/{captureName}" for captureName in os.listdir("Record/Motors")]
        vehicles += [f"Record/Vehicles/{captureName}" for captureName in os.listdir("Record/Vehicles")]
        
        return vehicles
        
    def searchVehicle(self):
        plateNumber = self.carPlateLabel.text()
        category = self.carCategoryComboBox.currentText()
        
        self.vehicles = self.readRecord()
        if category != "All":
            print(f"category: {category}")
            vehicles = []
            for vehicle in self.vehicles:
                if category in vehicle:
                    vehicles.append(vehicle)
            self.vehicles = vehicles
            
        if plateNumber != "":
            vehicles = []
            for vehicle in self.vehicles:
                if plateNumber in vehicle:
                    vehicles.append(vehicle)
            self.vehicles = vehicles
            
        self.page = 0
        self.showCapture(self.page)
        
    def turn_front_page(self):
        if self.page > 0:
            self.page -= 1
        self.showCapture(self.page)
        
    def turn_next_page(self):
        maxPage = (len(self.vehicles) - 1) / 15
        if self.page + 1 <= maxPage:
            self.page += 1
        self.showCapture(self.page)
        
        
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    
    mainWindow = SearchWidget()
    mainWindow.setStyleSheet("QLabel{font-size: 12pt;}")
    mainWindow.showMaximized()
    mainWindow.show()
    
    sys.exit(app.exec_())