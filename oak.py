#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np
import argparse
import time
import blobconverter

cam_options = ['rgb', 'left', 'right']

parser = argparse.ArgumentParser()
parser.add_argument("-cam", "--cam_input", help="select camera input source for inference", default='rgb', choices=cam_options)
parser.add_argument("-s", "--size", help="Shape", choices=['256','513'], default='256', type=str)
args = parser.parse_args()

cam_source = args.cam_input
nn_shape = int(args.size)

def decode_yolo11(output_tensor):
    # Assuming the YOLOv11 output tensor has the format [1, num_detections, 85] for segmentation
    output = output_tensor.reshape(-1, 85)  # Reshape depending on your blob
    boxes, confidences, class_ids = output[:, :4], output[:, 4], output[:, 5:]
    return boxes, confidences, class_ids

def show_yolo11(output_boxes, output_classes, frame):
    # Assuming class 0 is background and class 1 is the object of interest
    for i, box in enumerate(output_boxes):
        if output_classes[i] == 1:  # assuming class 1 is the object of interest
            x, y, w, h = map(int, box)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return frame

# Start defining a pipeline
pipeline = dai.Pipeline()
detection_nn = pipeline.create(dai.node.NeuralNetwork)
nn_path = blobconverter.from_zoo(name="yolov11_segmentation", zoo_type="depthai", shaves=6)  # Replace with your custom yoloseg.blob
detection_nn.setBlobPath(nn_path)

detection_nn.setNumPoolFrames(4)
detection_nn.input.setBlocking(False)
detection_nn.setNumInferenceThreads(2)

cam = None
# Define a source - color camera
if cam_source == 'rgb':
    cam = pipeline.create(dai.node.ColorCamera)
    cam.setPreviewSize(nn_shape, nn_shape)
    cam.setInterleaved(False)
    cam.preview.link(detection_nn.input)
elif cam_source == 'left':
    cam = pipeline.create(dai.node.MonoCamera)
    cam.setBoardSocket(dai.CameraBoardSocket.LEFT)
elif cam_source == 'right':
    cam = pipeline.create(dai.node.MonoCamera)
    cam.setBoardSocket(dai.CameraBoardSocket.RIGHT)

if cam_source != 'rgb':
    manip = pipeline.create(dai.node.ImageManip)
    manip.setResize(nn_shape, nn_shape)
    manip.setKeepAspectRatio(True)
    manip.setFrameType(dai.RawImgFrame.Type.BGR888p)
    cam.out.link(manip.inputImage)
    manip.out.link(detection_nn.input)

cam.setFps(40)

# Create outputs
