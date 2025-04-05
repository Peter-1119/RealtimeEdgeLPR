import torch
import torch.nn as nn
import numpy as np
import cv2
import time

class BidirectionalLSTM(nn.Module):
    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output


class CRNN(nn.Module):
    def __init__(self, imgH, nc, nclass, nh, n_rnn=2, leakyRelu=False):
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        convRelu(2, True)
        convRelu(3)
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16
        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16
        convRelu(6, True)  # 512x1x16

        self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nclass))

    def forward(self, input):
        # conv features
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        # rnn features
        output = self.rnn(conv)

        return output


class TextRecognizeEngine:
    def __init__(self, weight, alphabet = '0123456789abcdefghijklmnopqrstuvwxyz軍外使'):
        # detail refer: https://www.796t.com/article.php?id=298432
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = True
        
        self.alphabet = '-' + alphabet
        self.model = CRNN(32, 1, len(self.alphabet), 256)
        if torch.cuda.is_available(): self.model = self.model.cuda()
        self.model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(weight).items()})

    def decode(self, t, length):
        char_list = []
        for i in range(length):
            if t[i][0] != 0 and (not (i > 0 and t[i - 1][0] == t[i][0])):
                char_list.append(self.alphabet[t[i][0]])
        return ''.join(char_list)

    def decode_batch(self, t, batch, length):
        strings = []
        for b in range(batch):
            string = []
            for i in range(length):
                if t[i][b] != 0 and (not (i > 0 and t[i - 1][b] == t[i][b])):
                    string.append(self.alphabet[t[i][b]])
            strings.append((''.join(string)).upper())
        
        return strings

    # Detect text function
    def detect(self, img):
        crnn_start = time.time()
        sim_pred = 'Plate'
        
        crnn_pstart = time.time()
        image = cv2.resize(img, (100, 32), interpolation = cv2.INTER_NEAREST)
        # image = cv2.resize(cropImg, (100, 32), interpolation = cv2.INTER_AREA)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image.astype(np.float32) / 255.	
        image = (image - 0.5) * 2
        image = np.expand_dims(image, (0, 1))
        image = torch.Tensor(image).cuda() if torch.cuda.is_available() else torch.Tensor(image)
        self.crnn_ptime = time.time() - crnn_pstart

        with torch.no_grad():
            result = self.model(image).cpu().numpy()
        pred = np.argmax(result, axis = 2)
        sim_pred = self.decode(pred, len(pred)).upper()

        return sim_pred, time.time() - crnn_start

    def detect_batch(self, image, bbox):
        crnn_start = time.time()
        if len(bbox) == 0:
            return [], time.time() - crnn_start
        
        bbox[:, 0] = np.where(bbox[:, 0] < 0, 0, bbox[:, 0])
        bbox[:, 1] = np.where(bbox[:, 1] < 0, 0, bbox[:, 1])
        bbox[:, 2] = np.where(bbox[:, 2] >= image.shape[1], image.shape[1] - 1, bbox[:, 2])
        bbox[:, 3] = np.where(bbox[:, 3] >= image.shape[0], image.shape[0] - 1, bbox[:, 3])
        
        batch = bbox.shape[0]
        imgs = np.zeros((batch, 1, 32, 100))
        for index, (x1, y1, x2, y2, _) in enumerate(bbox):
            img = image[int(y1): int(y2), int(x1): int(x2)].copy()
            image_resized = cv2.resize(img, (100, 32), interpolation = cv2.INTER_NEAREST)
            # image = cv2.resize(cropImg, (100, 32), interpolation = cv2.INTER_AREA)
            image_gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
            image_normalized = image_gray.astype(np.float32) / 255.	
            image_normalized = (image_normalized - 0.5) * 2
            imgs[index][0] = image_normalized
        imgs = torch.Tensor(imgs).cuda() if torch.cuda.is_available() else torch.Tensor(image)

        with torch.no_grad():
            result = self.model(imgs).cpu().numpy()
        pred = np.argmax(result, axis = 2)
        sim_pred = self.decode_batch(pred, batch, len(pred))

        return sim_pred, time.time() - crnn_start