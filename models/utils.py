import numpy as np
import cv2

def open_cam_rtsp(uri, width, height, latency,buffer_size):                         #connect 192.168.0.53 camera
    gst_str = ('rtspsrc tcp-timeout=2000000 location={} latency={} ! '
               'application/x-rtp, media=video ! queue ! decodebin ! '
               'nvvidconv ! '
               'video/x-raw, width=(int){}, height=(int){}, '
               'format=(string)BGRx ! '
               'videoconvert ! appsink max-buffers={} drop=true').format(uri, latency, width, height,buffer_size)
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

def plot_one_box(x, im, color=(0,0,255), label=None, line_thickness=3):  
    '''
    plot the box
    '''
    tl = line_thickness or round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1  # line/font thickness
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(im, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(im, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

class obj_info(object):
    def __init__(self, obj_id, obj_pos, obj_cls, obj_time, obj_img):
        self.obj_id = obj_id
        self.obj_pos = obj_pos
        self.obj_cls = obj_cls
        self.obj_time = obj_time
        self.obj_img = obj_img

    def create_VA_info(self, binding_helmet, binding_plate):
        self.helmet_id = None
        self.plate_id = None
        self.motor_binding_helmet(binding_helmet)
        self.VA_binding_plate(binding_plate)
        self.plate = None

    def VA_binding_plate(self, binding):
        if self.obj_id in binding.keys():
            self.plate_id = binding[self.obj_id]

    def motor_binding_helmet(self, binding):
        if self.obj_id in binding.keys():
            self.helmet_id = binding[self.obj_id]

    def create_plate_info(self):
        self.plate = {}

    def plate_filter(self):
        plate_numbers = list(self.plate.keys())
        plate_numbers_count = list(self.plate.values())

        max_value_index = plate_numbers_count.index(max(plate_numbers_count))
        plate_number = plate_numbers[max_value_index]
        return plate_number
        
        
def LetterBox(Image, ContainerSize):
    '''
    Args:
        Image (H * W * C): Numpy formated image
        ContainerSize (List): (Width, Height)

    Returns:
        Resized numpy formated image

    '''
    ImageContainer = np.zeros((ContainerSize[1], ContainerSize[0], 3), dtype = np.uint8)
    H, W, C = Image.shape

    scaled_ratio_W = ContainerSize[1] / H
    scaled_ratio_H = ContainerSize[0] / W
    scaled_ratio = scaled_ratio_H if scaled_ratio_H < scaled_ratio_W else scaled_ratio_W
    
    New_H, New_W = int(H * scaled_ratio), int(W * scaled_ratio)

    dh, dw = int((ContainerSize[1] - New_H) / 2), int((ContainerSize[0] - New_W) / 2)

    # im0 = cv2.resize(im0, (New_W, New_H), interpolation = cv2.INTER_LINEAR)  # yolov5 origin zoom in code
    # im0 = cv2.resize(im0, (New_W, New_H), interpolation = cv2.INTER_AREA)  # Slow but Performance best
    ResizeImage = cv2.resize(Image, (New_W, New_H), interpolation = cv2.INTER_NEAREST)  # fast but performance worst
    
    ImageContainer[dh: New_H + dh, dw: New_W + dw] = ResizeImage
    return ImageContainer
