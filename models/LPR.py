import os
import cv2
import time
import numpy as np

from models.yoloEngine import yolov5_engine
from models.TextRecognize import TextRecognizeEngine
from models.utils import *

from PyQt5 import QtCore
from ObjectTracker.KF_tracking import Tracker
from ObjectTracker.TrackerDetection import *

# Check target object in ROI
def object_in_ROI(image_shape, center, ratio = 0.075):
    h, w, c = image_shape
    h1, h2 = h * ratio, h * (1 - ratio)
    w1, w2 = w * ratio, w * (1 - ratio)
    posX, posY = center
    if (h1 < posY < h2) and (w1 < posX < w2):
        return True
    return False

# Bind vehicle and attachments (like: helmet or license plate)
def bind_objects_(VA_boxes, attach_boxes, binding_type = "plate"):
    attach_binding = {}
    for VA_index, VA_box in enumerate(VA_boxes):
        VA_id, VA_alive_Frame, VA_x1, VA_y1, VA_x2, VA_y2, VA_label = VA_box
        VA_centerX, VA_centerY = (VA_x1 + VA_x2) / 2, (VA_y1 + VA_y2) / 2
        
        for attach_index, (x1, y1, x2, y2, category) in enumerate(attach_boxes):
            centerX, centerY = (x1 + x2) / 2, (y1 + y2) / 2
            
            if binding_type == "helmet" and (VA_x1 < centerX < VA_x2) and (VA_y1 < centerY < VA_y2):
                attach_binding[VA_index] = attach_index
            
            if binding_type == "plate" and (VA_x1 < centerX < VA_x2) and (VA_centerY < centerY < VA_y2):
                attach_binding[VA_index] = attach_index
    
    return attach_binding

# Bind attachments and vehicle
def bind_objects(VA_boxes, attach_boxes, binding_type = "plate"):
    attach_binding = {}
    for attach_index, (x1, y1, x2, y2, category) in enumerate(attach_boxes):
        centerX, centerY = (x1 + x2) / 2, (y1 + y2) / 2
        
        for VA_index, VA_box in enumerate(VA_boxes):
            VA_id, VA_alive_Frame, VA_x1, VA_y1, VA_x2, VA_y2, VA_label = VA_box
            VA_centerX, VA_centerY = (VA_x1 + VA_x2) / 2, (VA_y1 + VA_y2) / 2
            
            if binding_type == "helmet" and (VA_x1 < centerX < VA_x2) and (VA_y1 < centerY < VA_y2):
                attach_binding[VA_index] = attach_index
            
            if binding_type == "plate" and (VA_x1 < centerX < VA_x2) and (VA_centerY < centerY < VA_y2):
                attach_binding[VA_index] = attach_index
    
    return attach_binding
    
# Count the occurrences of recognized license plate texts.
def label_count(plate_info, label):
    if plate_info.get(label) == None:
        plate_info[label] = 1
    else:
        plate_info[label] += 1

# Select the most frequently recognized license plate text.
def choose_label(plate_info):
    label, labelCount = None, 0
    for plate, count in plate_info.items():
        if count > labelCount:
            label = plate
            labelCount = count
    
    return label

# Determine whether a helmet was detected in the majority of recent frames.
def check_helmet(helmet_info):
    if (sum(helmet_info) / len(helmet_info)) >= 0.5:
        return True
    else:
        return False

class LPR(QtCore.QThread):
    def __init__(self, yolo_engine, classes_txt, crnn_weight, attach_index, VA_index = None): # mode: [0,all] , [1,mask] , [2,temperature]
        super().__init__()
        
        # Load LPR model
        self.textRecognizeEngine = TextRecognizeEngine(crnn_weight)
        self.yoloEngine = yolov5_engine(640, yolo_engine)
        
        self.image = None
        self.start_play = False
        self.record_imgs_update = False
        
        # Declare tracker
        self.classNames = self.read_classes(classes_txt)
        self.VA_index = VA_index
        self.attach_index = attach_index
        self.VA_tracker = Tracker(40, 3, 15, 0)

        # Check folder for storage photo
        os.makedirs("Record/Vehicles", exist_ok = True)
        os.makedirs("Record/Motors", exist_ok = True)
        
        self.raise_signal = None
        self.capture_photos = []
        self.CarIDInfo = {}
    
    # Read class names from a YOLO model class file.
    def read_classes(self, filename):
        with open(filename, 'r') as f:
            classNames = [c[:-1] for c in f.readlines()]
            
        return classNames
    
    # Draw rectangle on detected objects
    def draw_box(self, image, bbox):
        for box in bbox:
            VA_id, VA_alive_Frame, x1, y1, x2, y2, category = box
            label = self.classNames[int(category)]
            
            if VA_alive_Frame > 5:
                plot_one_box([x1, y1, x2, y2], image, label = f"{int(VA_id)}_{label}")
    
    # Store detect vehicle and features and process
    def store_memory(self, image, VA_boxes, plate_boxes, plate_labels, helmet_relation, plate_relation):
        now = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        
        image_shape = image.shape
        
        # Correct vehicle bounding box
        VA_boxes[:, 2] = np.where(VA_boxes[:, 2] < 0, 0, VA_boxes[:, 2])
        VA_boxes[:, 3] = np.where(VA_boxes[:, 3] < 0, 0, VA_boxes[:, 3])
        VA_boxes[:, 4] = np.where(VA_boxes[:, 4] >= image_shape[1], image_shape[1] - 1, VA_boxes[:, 4])
        VA_boxes[:, 5] = np.where(VA_boxes[:, 5] >= image_shape[0], image_shape[0] - 1, VA_boxes[:, 5])
        
        # Correct license plate bounding box
        plate_boxes[:, 0] = np.where(plate_boxes[:, 0] < 0, 0, plate_boxes[:, 0])
        plate_boxes[:, 1] = np.where(plate_boxes[:, 1] < 0, 0, plate_boxes[:, 1])
        plate_boxes[:, 2] = np.where(plate_boxes[:, 2] >= image_shape[1], image_shape[1] - 1, plate_boxes[:, 2])
        plate_boxes[:, 3] = np.where(plate_boxes[:, 3] >= image_shape[0], image_shape[0] - 1, plate_boxes[:, 3])
        
        for VA_index, VA_box in enumerate(VA_boxes):
            VA_id, VA_alive_Frame, VA_x1, VA_y1, VA_x2, VA_y2, VA_label = VA_box
            VA_centerX, VA_centerY = (VA_x1 + VA_x2) / 2, (VA_y1 + VA_y2) / 2
            
            if object_in_ROI(image.shape, [VA_centerX, VA_centerY]):
                area = (VA_x2 - VA_x1) * (VA_y2 - VA_y1)
                # Add new car in storage
                if self.CarIDInfo.get(VA_id) == None:
                    capture = image[int(VA_y1):int(VA_y2), int(VA_x1):int(VA_x2)]
                    self.CarIDInfo[VA_id] = {"category": VA_label, "alive": VA_alive_Frame, "capture": capture, 'capture_size': area, 'capture_time': now, "plate": {}, "helmet": []}
                else:
                    self.CarIDInfo[VA_id]['alive'] = VA_alive_Frame
                    
                # Record car plate
                if plate_relation.get(VA_index) != None:
                    plate_box = plate_boxes[plate_relation[VA_index]]
                    x1, y1, x2, y2, category = plate_box
                    # label, crnn_time = self.textRecognizeEngine.detect(image[int(y1):int(y2), int(x1):int(x2)])
                    label_count(self.CarIDInfo[VA_id]['plate'], plate_labels[plate_relation[VA_index]])
                
                # Record helmet event
                if helmet_relation.get(VA_index) != None:
                    self.CarIDInfo[VA_id]["helmet"].append(True)
                else:
                    self.CarIDInfo[VA_id]["helmet"].append(False)
                    
                # Update capture
                if area > self.CarIDInfo[VA_id]['capture_size']:
                    capture = image[int(VA_y1):int(VA_y2), int(VA_x1):int(VA_x2)]
                    self.CarIDInfo[VA_id]['capture'] = capture
                    self.CarIDInfo[VA_id]['capture_size'] = area
                    self.CarIDInfo[VA_id]['capture_time'] = now
        
    # Capture and save photos of vehicles for later searching
    def capture_photo(self, delete_ids):
        self.capture_photos.clear()
        for delete_id in delete_ids:
            if self.CarIDInfo.get(delete_id) != None:
                VA_category = self.classNames[int(self.CarIDInfo[delete_id]['category'])]
                VA_alive = self.CarIDInfo[delete_id]['alive']
                VA_time = self.CarIDInfo[delete_id]['capture_time']
                VA_photo = self.CarIDInfo[delete_id]['capture']
                plate_label = choose_label(self.CarIDInfo[delete_id]['plate'])
                helmet_check = check_helmet(self.CarIDInfo[delete_id]['helmet'])
                
                if VA_alive > 5:
                    if VA_category == 'motor':
                        print(f"save: {VA_category} - {plate_label}")
                        cv2.imwrite(f"Record/Motors/{delete_id}_{VA_time}_{VA_category}_{plate_label}_{helmet_check}.jpg", VA_photo)
                        self.capture_photos.append(VA_photo)
                    
                    else:
                        print(f"save: {VA_category} - {plate_label}")
                        cv2.imwrite(f"Record/Vehicles/{delete_id}_{VA_time}_{VA_category}_{plate_label}.jpg", VA_photo)
                        self.capture_photos.append(VA_photo)
                
                del self.CarIDInfo[delete_id]
    
    def detect_and_recognize(self, image):
        # Yolo model detect objects
        result = self.yoloEngine.predict(image)
        
        image_shape = image.shape
        
        # Calculate and track vehicle features
        VA_class = [c in self.VA_index for c in result[:, 4]]
        VA_centers = calc_center_without_label(result[VA_class])
        VA_boxes, VA_delete_ids = get_tracking_object(self.VA_tracker, VA_centers, result[VA_class])
        
        # Select attachment from detected objects
        helmet_boxes = result[result[:, 4] == self.attach_index[0]]
        plate_boxes = result[result[:, 4] == self.attach_index[1]]
        
        # Bind attachment and vehicle
        helmet_relation = bind_objects(VA_boxes, helmet_boxes, binding_type = "helmet")
        plate_relation = bind_objects(VA_boxes, plate_boxes, "plate")
        plate_labels, _ = self.textRecognizeEngine.detect_batch(image, plate_boxes)
        
        # Store and process vehicle feature, and then capture photo for disappear object
        if VA_boxes.shape[0] != 0:
            self.store_memory(image.copy(), VA_boxes, plate_boxes, plate_labels, helmet_relation, plate_relation)
        self.capture_photo(VA_delete_ids)
        
        # Draw rectangle for detected objects
        self.draw_box(image, VA_boxes)
        
        # Return frame
        return image
    
    def run(self):
        cap1 = cv2.VideoCapture('videos/test.mkv')
        index = 0
        
        while cap1.isOpened():
            # start = time.time()
            
            # Read frame from cap instance
            ret0, im0 = cap1.read()  # origin image
            if not ret0: break
            
            # Process frame
            image = self.detect_and_recognize(im0)
            
            # Send capture and current frame to interface
            if self.raise_signal != None:
                self.raise_signal.emit({"video": image, "captures": self.capture_photos})
            
            # Show processed frame
            # cv2.imshow('image_win', image)
            
            # key = cv2.waitKey(1)
            # if key == ord('q'):
            #     print("Exit...")
            #     break

            # print('##### spend time: {}#####'.format(time.time() - start))
        
        cv2.destroyAllWindows()
        cap1.release()
        
if __name__ == '__main__':
    save_classes_no_normal_car = [0, 1, 2, 5, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
    attach_index = [2, 8]
    lpr = LPR("models/weights/VA3/best.engine", "models/classes.txt", "models/weights/crnn/BestNetCN.pth", attach_index, save_classes_no_normal_car)
    lpr.run()