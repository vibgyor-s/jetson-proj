#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt
import threading
import cv2
from stereovision.calibration import StereoCalibration


#Class that helps in starting camera
class Start_Cameras:

    def __init__(self, sensor_id):

        # Initialize instance variables
        # OpenCV video capture element

        self.video_capture = None

        # The last captured image from the camera

        self.frame = None
        self.grabbed = False

        # The thread where the video capture runs

        self.read_thread = None
        self.read_lock = threading.Lock()
        self.running = False

        self.sensor_id = sensor_id

        gstreamer_pipeline_string = self.gstreamer_pipeline()
        self.open(gstreamer_pipeline_string)

    # Opening the cameras

    def open(self, gstreamer_pipeline_string):
        gstreamer_pipeline_string = self.gstreamer_pipeline()
        try:
            self.video_capture = \
                cv2.VideoCapture(gstreamer_pipeline_string,
                                 cv2.CAP_GSTREAMER)
            (grabbed, frame) = self.video_capture.read()
            print('Cameras are opened')
        except RuntimeError:

            self.video_capture = None
            print('Unable to open camera')
            print('Pipeline: ' + gstreamer_pipeline_string)
            return

        # Grab the first frame to start the video capturing

        (self.grabbed, self.frame) = self.video_capture.read()

    # Starting the cameras

    def start(self):
        if self.running:
            print('Video capturing is already running')
            return None

        # create a thread to read the camera image

        if self.video_capture != None:
            self.running = True
            self.read_thread = \
                threading.Thread(target=self.updateCamera, daemon=True)
            self.read_thread.start()
        return self

    def stop(self):
        self.running = False
        self.read_thread.join()

    def updateCamera(self):

        # This is the thread to read images from the camera

        while self.running:
            try:
                (grabbed, frame) = self.video_capture.read()
                with self.read_lock:
                    self.grabbed = grabbed
                    self.frame = frame
            except RuntimeError:
                print('Could not read image from camera')

    def read(self):
        with self.read_lock:
            frame = self.frame.copy()
            grabbed = self.grabbed
        return (grabbed, frame)

    def release(self):
        if self.video_capture != None:
            self.video_capture.release()
            self.video_capture = None

        # Now kill the thread

        if self.read_thread != None:
            self.read_thread.join()

    # Currently there are setting frame rate on CSI Camera on Nano through gstreamer
    # Here we directly select sensor_mode 3 (1280x720, 59.9999 fps)

    def gstreamer_pipeline(
        self,
        sensor_mode=3,
        capture_width=1280,
        capture_height=720,
        display_width=640,
        display_height=360,
        framerate=30,
        flip_method=0,
        ):

        return 'nvarguscamerasrc sensor-id=%d sensor-mode=%d ! video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, format=(string)NV12, framerate=(fraction)%d/1 ! nvvidconv flip-method=%d ! video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink' \
            % (
            self.sensor_id,
            sensor_mode,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
            )

#Camera started.
left_camera = Start_Cameras(0).start()
right_camera = Start_Cameras(1).start()

def nothing(x):
    pass

kernel = np.ones((2,2),np.uint8)

window_size = 13
min_disp =16
num_disp = 128-min_disp
stereo = cv2.StereoSGBM_create(
    minDisparity = min_disp,
    numDisparities = num_disp,
    blockSize = window_size,
    uniquenessRatio = 10,
    speckleWindowSize = 100,
    speckleRange = 32,
    disp12MaxDiff = 1,
    P1 = 8*3*window_size**2,
    P2 = 32*3*window_size**2,
)

while True:
    left_grabbed, left_frame = left_camera.read()
    right_grabbed, right_frame = right_camera.read()

    if left_grabbed and right_grabbed:
#        cv2.imshow("Left", left_frame)
#        cv2.imshow("Right", right_frame)

#        left_frame = cv2.flip(left_frame,-1)
#        right_frame = cv2.flip(right_frame,-1)


        left_frame_new = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
        right_frame_new = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)


        calibration = StereoCalibration(input_folder='../calib_result')
        rectified_pair = calibration.rectify((left_frame_new, right_frame_new))

        left_frame_new = rectified_pair[0]
        right_frame_new = rectified_pair[1]


#        stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
        #disparity = stereo.compute(left_frame_new,right_frame_new).astype(np.float32) / 16.0
#       disparity = (disparity-min_disp)/num_disp
#        disparity_normalized = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)

#        disparity_normalized = cv2.normalize(disparity, None, alpha = 0, beta = 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
#        threshold = cv2.threshold(disparity_normalized, 0.6, 1.0, cv2.THRESH_BINARY)[1]

        disparity = stereo.compute(left_frame_new,right_frame_new).astype(np.float32) / 16.0
        disparity = (disparity-min_disp)/num_disp
       # disparity = disparity.astype(np.float32)
        #disparity = (disparity/16.0 - minDisparity)/numDisparities

        
        disparity_normalized = cv2.normalize(disparity, None, alpha = 0, beta = 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        threshold = cv2.threshold(disparity_normalized, 0.9, 1.0, cv2.THRESH_BINARY)[1]
        morphology = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)


#        contours, hierarchy = cv2.findContours(image=threshold, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
#        cv.drawContours(threshold, contours, -1, (0,255,0), 3)





        cv2.imshow('disparity',morphology)

        k = cv2.waitKey(1) & 0xFF

        if k == ord('q'):
            break
    else:
        continue

left_camera.stop()
left_camera.release()
right_camera.stop()
right_camera.release()
cv2.destroyAllWindows()