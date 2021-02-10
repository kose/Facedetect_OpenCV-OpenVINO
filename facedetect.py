# -*- coding: utf-8 -*-

import argparse
import cv2 as cv

def facedetect(device, threshold):

    # load model
    net = cv.dnn.readNet('face-detection-adas-0001.xml',
                         'face-detection-adas-0001.bin')

    # input size of Neural Nwtwork
    inputsize = (672, 384)

    # output name
    outNames = cv.dnn_Net.getLayerNames(net)

    # select backend
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_INFERENCE_ENGINE)

    # select inference device
    if device == 'MYRIAD':
        net.setPreferableTarget(cv.dnn.DNN_TARGET_MYRIAD)
    elif device == 'CPU':
        net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
    else:
        print("Unkown device: " + device)
        exit(0)

    # capture device
    cap = cv.VideoCapture(0)

    # set capcture image size
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 800)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 600)

    while cv.waitKey(16) != 27:

        _, frame = cap.read()

        # create a 4D blob from a frame.
        blob = cv.dnn.blobFromImage(frame, size=inputsize, ddepth=cv.CV_8U)

        # set the input to the network
        net.setInput(blob)

        # inference
        out = net.forward(outNames[0])

        # [1, 1, N, 7] -> [N, 7]
        for detection in out.reshape(-1, 7):
            confidence = float(detection[2])

            if confidence > threshold:
                xmin = int(detection[3] * frame.shape[1])
                ymin = int(detection[4] * frame.shape[0])
                xmax = int(detection[5] * frame.shape[1])
                ymax = int(detection[6] * frame.shape[0])
                cv.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        cv.imshow('facedetect', frame)


def main():
    parser = argparse.ArgumentParser(description='Facedetect with OpenCV DNN API (OpenVINO)')
    parser.add_argument('--device', '-d', default="CPU")
    parser.add_argument('--threshold', '-t', type=float, default=0.9)
    args = parser.parse_args()

    facedetect(args.device, args.threshold)

if __name__ == '__main__':
    main()

# import pdb; pdb.set_trace()

### Local Variables: ###
### truncate-lines:t ###
### End: ###
