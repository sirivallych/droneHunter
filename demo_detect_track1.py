# RGB-1.1--only rgb
from __future__ import division  # Must be first
import sys
import os
import time
import warnings
warnings.filterwarnings("ignore")

# -------------------- Add parent folder to sys.path --------------------
project_root = os.path.dirname(os.path.abspath(__file__))  # Codes folder
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# -------------------- Add YOLOv5 folder to sys.path --------------------
yolov5_path = os.path.join(project_root, 'yolov5')
if yolov5_path not in sys.path:
    sys.path.insert(0, yolov5_path)

# -------------------- Add Tracking Wrapper folders --------------------
tracking_wrapper_path1 = os.path.join(project_root, 'tracking_wrapper', 'dronetracker')
tracking_wrapper_path2 = os.path.join(project_root, 'tracking_wrapper', 'drtracker')
if tracking_wrapper_path1 not in sys.path:
    sys.path.append(tracking_wrapper_path1)
if tracking_wrapper_path2 not in sys.path:
    sys.path.append(tracking_wrapper_path2)

# -------------------- Standard Imports --------------------
import queue
import pdb
import datetime
import argparse
import struct
import socket
import json
from PIL import Image
import cv2
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
from collections import deque
import numpy as np

# -------------------- Custom Imports --------------------
from detect_wrapper.Detectoruav import DroneDetection
from detect_wrapper.utils.datasets import LoadStreams, LoadImages
from tracking_wrapper.dronetracker.trackinguav.evaluation.tracker import Tracker

# -------------------- Video Path & Settings --------------------
video_path = os.path.join(project_root, 'testvideo', 'a3.mp4')
magnification = 1  # ✅ Keep original video size

# -------------------- Helper Function --------------------
def lefttop2center(bbx):
    obbx = [0, 0, bbx[2], bbx[3]]
    obbx[0] = bbx[0] + bbx[2] / 2
    obbx[1] = bbx[1] + bbx[3] / 2
    return obbx

def bbox_is_valid(bbx, frame_shape):
    """Check if bounding box is valid for tracking"""
    if bbx is None:
        return False
    try:
        x, y, w, h = [int(v) for v in bbx]
        if w <= 0 or h <= 0:
            return False
        H, W = frame_shape[:2]
        if x >= W or y >= H:
            return False
        # allow partial out-of-bounds but not completely out
        if x + w <= 0 or y + h <= 0:
            return False
        return True
    except (ValueError, TypeError, IndexError):
        return False

# -------------------- Main Test Function --------------------
def test():
    # IR weights (you can leave as is)
    IRweights_path = os.path.join(project_root, 'detect_wrapper', 'weights', 'best.pt')
    
    # RGB-trained YOLOv5 weights (your new model)
    RGBweights_path = os.path.join(project_root, 'detect_wrapper', 'weights', 'drone_rgb_yolov5s.pt')
    # RGBweights_path = os.path.join(os.path.dirname(project_root), 'checkpoints', 'drone_rgb_yolov5s.pt')
    
    time_record = []
    det_time = []
    interval = 50
    cap = cv2.VideoCapture(video_path)
    
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read video file.")
        return

    oframe = frame.copy()
    # Pass both IR and RGB weights to DroneDetection
    drone_det = DroneDetection(IRweights_path=IRweights_path, RGBweights_path=RGBweights_path)
    drone_tracker = Tracker()
    first_track = True

    exit_flag = False
    while ret and not exit_flag:
        t1 = time.time()
        # ✅ Use RGB detection instead of IR
        detections = drone_det.forward_RGB(frame)
        t2 = time.time()
        det_time.append(t2 - t1)

        # ✅ No resizing; display original frame size
        visuframe = oframe.copy()
        
        # Process all detections
        valid_detections = []
        for init_box, det_conf in detections:
            if det_conf > 0.5 and bbox_is_valid(init_box, frame.shape):
                valid_detections.append((init_box, det_conf))
                bbx = [int(x) for x in init_box]
                print(f"Detection BB: {bbx} (conf: {det_conf:.2f})")
                
                # Draw bounding box for each detection
                cv2.rectangle(visuframe, (bbx[0], bbx[1]), 
                              (bbx[0] + bbx[2], bbx[1] + bbx[3]), (0, 255, 0), 2)
                cv2.putText(visuframe, f'Drone {det_conf:.2f}', (bbx[0], bbx[1] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Use the highest confidence detection for tracking
        if valid_detections:
            # Sort by confidence and use the best one for tracking
            best_detection = max(valid_detections, key=lambda x: x[1])
            init_box, det_conf = best_detection
            
            if first_track:
                drone_tracker.init_track(init_box, frame)
                first_track = False
            else:
                drone_tracker.change_state(init_box)
        else:
            # Add "No Drone Detected" caption when no drone is found
            visuframe = oframe.copy()
            text = "No Drone Detected"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            color = (0, 0, 255)  # Red color
            thickness = 2
            
            # Get text size for centering
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            frame_height, frame_width = visuframe.shape[:2]
            
            # Center the text
            x = (frame_width - text_width) // 2
            y = (frame_height + text_height) // 2
            
            # Add background rectangle for better visibility
            padding = 10
            cv2.rectangle(visuframe, 
                        (x - padding, y - text_height - padding), 
                        (x + text_width + padding, y + baseline + padding), 
                        (255, 255, 255), -1)
            
            # Add the text
            cv2.putText(visuframe, text, (x, y), font, font_scale, color, thickness)
            cv2.imshow("tracking", visuframe)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC to quit
                exit_flag = True
                break

        # Display the current frame
        cv2.imshow("tracking", visuframe)
        key = cv2.waitKey(30) & 0xFF  # Increased wait time for better video playback
        if key == ord('q') or key == 27:  # 'q' or ESC to quit
            exit_flag = True
            break

        # Read next frame for main loop
        ret, frame = cap.read()
        if not ret:
            break
        oframe = frame.copy()

        # Only do tracking if we have a valid tracker initialized
        if not first_track and not exit_flag:
            num = 0
            while num < interval and not exit_flag:
                num += 1
                ret, frame = cap.read()
                if not ret:
                    exit_flag = True
                    break
                oframe = frame.copy()
                visuframe = oframe.copy()
                t1 = time.time()
                try:
                    outputs = drone_tracker.on_track(frame)
                    t2 = time.time()
                    time_record.append(t2 - t1)

                    bbx = [int(i) for i in outputs]
                    if bbox_is_valid(bbx, frame.shape):
                        print("Tracking BB:", bbx)
                        cv2.rectangle(visuframe, (bbx[0], bbx[1]),
                                      (bbx[0] + bbx[2], bbx[1] + bbx[3]), (0, 255, 0), 2)
                        cv2.putText(visuframe, 'Drone', (bbx[0], bbx[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    else:
                        print("Invalid tracking BB, breaking tracking loop")
                        # Show "No Drone Detected" when tracking fails
                        text = "No Drone Detected"
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 1.0
                        color = (0, 0, 255)  # Red color
                        thickness = 2
                        
                        # Get text size for centering
                        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                        frame_height, frame_width = visuframe.shape[:2]
                        
                        # Center the text
                        x = (frame_width - text_width) // 2
                        y = (frame_height + text_height) // 2
                        
                        # Add background rectangle for better visibility
                        padding = 10
                        cv2.rectangle(visuframe, 
                                    (x - padding, y - text_height - padding), 
                                    (x + text_width + padding, y + baseline + padding), 
                                    (255, 255, 255), -1)
                        
                        # Add the text
                        cv2.putText(visuframe, text, (x, y), font, font_scale, color, thickness)
                        break
                except Exception as e:
                    print(f"Tracking error: {e}")
                    # Show "No Drone Detected" when tracking fails
                    text = "No Drone Detected"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1.0
                    color = (0, 0, 255)  # Red color
                    thickness = 2
                    
                    # Get text size for centering
                    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                    frame_height, frame_width = visuframe.shape[:2]
                    
                    # Center the text
                    x = (frame_width - text_width) // 2
                    y = (frame_height + text_height) // 2
                    
                    # Add background rectangle for better visibility
                    padding = 10
                    cv2.rectangle(visuframe, 
                                (x - padding, y - text_height - padding), 
                                (x + text_width + padding, y + baseline + padding), 
                                (255, 255, 255), -1)
                    
                    # Add the text
                    cv2.putText(visuframe, text, (x, y), font, font_scale, color, thickness)
                    break
                cv2.imshow("tracking", visuframe)
                key = cv2.waitKey(30) & 0xFF  # Increased wait time for better video playback
                if key == ord('q') or key == 27:  # 'q' or ESC to quit
                    exit_flag = True
                    break

    cap.release()
    cv2.destroyAllWindows()
    print("Done......")
    print('Track average time:', np.array(time_record).mean())
    print('Detect average time:', np.array(det_time).mean())

# -------------------- Run --------------------
if __name__ == "__main__":
    test()
