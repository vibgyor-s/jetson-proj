#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import cv2

import threading


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


CamL = Start_Cameras(0).start()
CamR = Start_Cameras(1).start()

# Reading the mapping values for stereo image rectification

cv_file = cv2.FileStorage('data/stereo_rectify_maps.xml',
                          cv2.FILE_STORAGE_READ)
Left_Stereo_Map_x = cv_file.getNode('Left_Stereo_Map_x').mat()
Left_Stereo_Map_y = cv_file.getNode('Left_Stereo_Map_y').mat()
Right_Stereo_Map_x = cv_file.getNode('Right_Stereo_Map_x').mat()
Right_Stereo_Map_y = cv_file.getNode('Right_Stereo_Map_y').mat()
cv_file.release()


def nothing(x):
    pass


cv2.namedWindow('disp', cv2.WINDOW_NORMAL)
cv2.resizeWindow('disp', 600, 600)

cv2.createTrackbar('numDisparities', 'disp', 1, 17, nothing)
cv2.createTrackbar('blockSize', 'disp', 5, 50, nothing)
cv2.createTrackbar('preFilterType', 'disp', 1, 1, nothing)
cv2.createTrackbar('preFilterSize', 'disp', 2, 25, nothing)
cv2.createTrackbar('preFilterCap', 'disp', 5, 62, nothing)
cv2.createTrackbar('textureThreshold', 'disp', 10, 100, nothing)
cv2.createTrackbar('uniquenessRatio', 'disp', 15, 100, nothing)
cv2.createTrackbar('speckleRange', 'disp', 0, 100, nothing)
cv2.createTrackbar('speckleWindowSize', 'disp', 3, 25, nothing)
cv2.createTrackbar('disp12MaxDiff', 'disp', 5, 25, nothing)
cv2.createTrackbar('minDisparity', 'disp', 5, 25, nothing)

# Creating an object of StereoBM algorithm

stereo = cv2.StereoBM_create()

while True:

    # Capturing and storing left and right camera images

    (retL, imgL) = CamL.read()
    (retR, imgR) = CamR.read()

    # retL, imgL= CamL.read()
    # retR, imgR= CamR.read()

    # Proceed only if the frames have been captured

    if retL and retR:
        imgR_gray = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
        imgL_gray = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)

        # Applying stereo image rectification on the left image

        # Left_nice = cv2.remap(
        #     imgL_gray,
        #     Left_Stereo_Map_x,
        #     Left_Stereo_Map_y,
        #     cv2.INTER_LANCZOS4,
        #     cv2.BORDER_CONSTANT,
        #     0,
        #     )

        # # Applying stereo image rectification on the right image

        # Right_nice = cv2.remap(
        #     imgR_gray,
        #     Right_Stereo_Map_x,
        #     Right_Stereo_Map_y,
        #     cv2.INTER_LANCZOS4,
        #     cv2.BORDER_CONSTANT,
        #     0,
        #     )

        # Updating the parameters based on the trackbar positions

        numDisparities = cv2.getTrackbarPos('numDisparities', 'disp') \
            * 16
        blockSize = cv2.getTrackbarPos('blockSize', 'disp') * 2 + 5
        preFilterType = cv2.getTrackbarPos('preFilterType', 'disp')
        preFilterSize = cv2.getTrackbarPos('preFilterSize', 'disp') * 2 \
            + 5
        preFilterCap = cv2.getTrackbarPos('preFilterCap', 'disp')
        textureThreshold = cv2.getTrackbarPos('textureThreshold', 'disp'
                )
        uniquenessRatio = cv2.getTrackbarPos('uniquenessRatio', 'disp')
        speckleRange = cv2.getTrackbarPos('speckleRange', 'disp')
        speckleWindowSize = cv2.getTrackbarPos('speckleWindowSize',
                'disp') * 2
        disp12MaxDiff = cv2.getTrackbarPos('disp12MaxDiff', 'disp')
        minDisparity = cv2.getTrackbarPos('minDisparity', 'disp')

        # Setting the updated parameters before computing disparity map

        stereo.setNumDisparities(numDisparities)
        stereo.setBlockSize(blockSize)
        stereo.setPreFilterType(preFilterType)
        stereo.setPreFilterSize(preFilterSize)
        stereo.setPreFilterCap(preFilterCap)
        stereo.setTextureThreshold(textureThreshold)
        stereo.setUniquenessRatio(uniquenessRatio)
        stereo.setSpeckleRange(speckleRange)
        stereo.setSpeckleWindowSize(speckleWindowSize)
        stereo.setDisp12MaxDiff(disp12MaxDiff)
        stereo.setMinDisparity(minDisparity)

        # Calculating disparity using the StereoBM algorithm

        disparity = stereo.compute(imgL,imgR)

        # NOTE: Code returns a 16bit signed single channel image,
        # CV_16S containing a disparity map scaled by 16. Hence it
        # is essential to convert it to CV_32F and scale it down 16 times.

        # Converting to float32

        disparity = disparity.astype(np.float32)

        # Scaling down the disparity values and normalizing them

        disparity = (disparity / 16.0 - minDisparity) / numDisparities

        # Displaying the disparity map

        cv2.imshow('disp', disparity)

        # Close window using esc key

        if cv2.waitKey(1) == 27:
            break
    else:
        break

CamL.stop()
CamL.release()
CamR.stop()
CamR.release()
cv2.destroyAllWindows()

