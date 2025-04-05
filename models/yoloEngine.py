import numpy as np
import cv2
import time
from models.Tensorrtx import *

class yolov5_engine:
    def __init__(self, img_size, weights):
        trt_engine_path = weights  # weight path "yolo.engine"
        self.model = TRTInference(trt_engine_path, 1)
        # self.model = TrtModel(trt_engine_path)
        # shape = self.model.engine.get_binding_shape(0)

        self.img_size = img_size
        self.yolo_time = 0
        self.yolo_ptime = 0
        self.yolo_atime = 0

    def letter_box(self, im0):
        new_im = np.zeros((self.img_size, self.img_size, 3))
        H, W, C = im0.shape

        self.scaled_ratio = self.img_size / H if H > W else self.img_size / W
        New_H, New_W = int(H * self.scaled_ratio), int(W * self.scaled_ratio)


        # im0 = cv2.resize(im0, (New_W, New_H), interpolation = cv2.INTER_LINEAR)  # yolov5 origin zoom in code
        # im0 = cv2.resize(im0, (New_W, New_H), interpolation = cv2.INTER_AREA)  # Slow but Performance best
        im0 = cv2.resize(im0, (New_W, New_H), interpolation = cv2.INTER_NEAREST)  # fast but performance worst

        new_im[: New_H, : New_W] = im0
        return new_im

    def transfer_img(self, im0, path = False):
        if path: im0 = cv2.imread(im0)
        yolo_tstart = time.time()

        im1 = self.letter_box(im0)
        self.dw, self.dh = 0, 0
        # im1, ratio, (self.dw, self.dh) = self.letter_box2(im0)

        im1 = im1.astype(np.float32) / 255.
        im1 = im1.transpose((2, 0, 1))[::-1]
        # print('time:', time.time() - yolo_tstart)

        return im0, im1

    def predict(self, img, path = False, verbose = False):
        yolo_pstart = time.time()
        self.im0, self.im1 = self.transfer_img(img, path)

        result, infer_time = self.model.infer(self.im1)
        if verbose: print('infer_time: {}'.format(infer_time))
        
        result, statistic_time = self.yolo_output(result)
        if verbose: print('statistic result: {}, statistic time: {}'.format(result.shape, statistic_time))
        
        return result[:, [0, 1, 2, 3, 5]]

    def xywh2xyxy(self, x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y

    def scaled(self, coords):
        h0, w0, c0 = self.im0.shape
        c1, h1, w1 = self.im1.shape

        ratio_W = w0 / w1
        ratio_H = h0 / h1

        coords[:, 0] = (coords[:, 0] - self.dw) / self.scaled_ratio
        coords[:, 1] = (coords[:, 1] - self.dh) / self.scaled_ratio
        coords[:, 2] = (coords[:, 2] - self.dw) / self.scaled_ratio
        coords[:, 3] = (coords[:, 3] - self.dh) / self.scaled_ratio
        return coords

    def NMS(self, dets, thresh):
        NMS_time = time.time()
        # assign x1, x2, y1, y2, score
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]


        # Calculate area and sort based on score
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        temp = []
        while order.size > 0:
            i = order[0]
            temp.append(i)
            
            # Get intersection over union ratio
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
    
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
    
            # Get rid of high IoU ratio
            inds = np.where(ovr <= thresh)[0]
            
            # Extract exclusive score
            order = order[inds + 1]
        
        return temp

    def yolo_output(self, result, conf_thres = 0.4, iou_thres = 0.45, max_wh = 1024):
        start = time.time()
        
        xc = result[..., 4] > conf_thres
        classes = result.shape[-1] - 5
        
        result = result[xc]
        if len(result) == 0: 
            return np.zeros((0, 6)), time.time() - start
        result[:, 5:] *= result[:, 4:5]

        box = self.xywh2xyxy(result[:, :4])
        conf = result[:, 4:5]
        max_index = np.argmax(result[:, 5:], 1)
        result = np.c_[box, conf, max_index]

        c = result[:, 5:6] * max_wh  # classes
        boxes, scores = result[:, :4] + c, result[:, 4]  # boxes (offset by class), scores

        temp = self.NMS(np.c_[boxes, scores], iou_thres)
        
        scaled_output = self.scaled(result[temp])
        return scaled_output, time.time() - start
