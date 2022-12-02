"""
Copyright (c) 2021-2022 UCLouvain, ICTEAM
Licensed under GPL-3.0 [see LICENSE for details]
Written by Jonathan Samelson (2021-2022)
"""

import json
import time
from timeit import default_timer
from tqdm import tqdm

import cv2
import ffmpeg
import numpy as np
import logging

from pytb.detection.detection_manager import DetectionManager
from pytb.detection.detector_factory import DetectorFactory
from pytb.tracking.tracker_factory import TrackerFactory
from pytb.tracking.tracking_manager import TrackingManager
from pytb.utils.video_capture_async import VideoCaptureAsync

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(10000, 3), dtype="uint8")
font = cv2.FONT_HERSHEY_DUPLEX
line_type = cv2.LINE_AA
thickness = 2


def main(cfg_detect, cfg_track, cfg_classes, video_path):

    log = logging.getLogger("aptitude-toolbox")

    # Get parameters of the different stages of the detection process
    with open(cfg_detect) as config_file:
        detect1 = json.load(config_file)
        log.debug("Detector config loaded.")

    detect1_proc = detect1['proc']
    detect1_preproc = detect1['preproc']
    detect1_postproc = detect1['postproc']

    # Get parameters of the different stages of the tracking process
    with open(cfg_track) as config_file:
        track1 = json.load(config_file)
        log.debug("Tracker config loaded.")

    track1_proc = track1['proc']
    track1_preproc = track1['preproc']
    track1_postproc = track1['postproc']

    # Get the classes of the object to be detected
    with open(cfg_classes) as config_file:
        CLASSES = json.load(config_file)['classes']
        log.debug("Classes config loaded.")

    # Instantiate the detector
    start = default_timer()
    detection_manager = DetectionManager(DetectorFactory.create_detector(detect1_proc), detect1_preproc,
                                         detect1_postproc)
    end = default_timer()
    log.info("Detector init duration = {}s".format(str(end - start)))

    # Instantiate the tracker
    start = default_timer()
    tracking_manager = TrackingManager(TrackerFactory.create_tracker(track1_proc), track1_preproc, track1_postproc)
    end = default_timer()
    log.info("Tracker init duration = {}s".format(str(end - start)))

    # If async flag is present, use the VideoCaptureAsync class
    # It allows to load the images in parallel of the detection/tracking process
    cap = cv2.VideoCapture(video_path)

    # Measure elapsed time to read the image
    read_time_start = default_timer()
    is_reading, frame = cap.read()
    if not is_reading:
        print("video not reading")
    read_time = default_timer() - read_time_start
    if is_reading:
        log.debug("Video file opened successfully.")
    else:
        log.error("Error while opening video file, check if the input path is correct.")

    is_paused = False

    # If record flag is present, initializes the VideoWriter to save the results of the process
    H, W, _ = frame.shape

    start_time = default_timer()
    last_update = default_timer()
    before_loop = start_time
    counter = 0
    tot_det_time = 0
    tot_track_time = 0

    output_lines = []

    # Get the number of frames of the video
    nb_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=nb_frames)
    

    while is_reading:

        # Check if a key was pressed but with some delay, as it is resource consuming
        time_update = default_timer()
        if time_update - last_update > (1/10):
            k = cv2.waitKey(1) & 0xFF
            if k == ord('p'):  # pause/play loop if 'p' key is pressed
                log.debug("Process paused/resumed.")
                is_paused = not is_paused
            if k == ord('q'):  # end video loop if 'q' key is pressed
                log.info("Process exited.")
                break
            last_update = time_update

        if is_paused:
            time.sleep(0.5)
            continue

        #detection and track
        (H, W, _) = frame.shape
        log.debug("Before detection.")
        det = detection_manager.detect(frame)
        log.debug("After detection & before tracking.")
        if tracking_manager.tracker.need_frame:
            res = tracking_manager.track(det, frame)
            log.debug("After tracking, with frame.")
        else:
            res = tracking_manager.track(det)
            log.debug("After tracking, without frames.")

        tot_det_time += det.detection_time
        tot_track_time += res.tracking_time

        # Change dimensions of the result to match to the initial dimension of the frame
        res.change_dims(W, H)
        log.debug("Dimensions of the results changed: (W: {}, H:{}).".format(W, H))



        res.to_x1_y1_x2_y2()
        log.debug("Results converted to x1,y1,x2,y2.")
        # print(res)

        
        # Add the bboxes from the process to the frame
        for i in range(res.number_objects):
            id = res.global_IDs[i]
            color = [int(c) for c in COLORS[id]]
            vehicle_label = 'I: {0}, T: {1} ({2})'.format(id, CLASSES[res.class_IDs[i]], str(res.det_confs[i])[:4])

            # Draw a rectangle (with a random color) for each bbox
            cv2.rectangle(frame, (round(res.bboxes[i][0]), round(res.bboxes[i][1])),
                            (round(res.bboxes[i][2]), round(res.bboxes[i][3])), color, thickness)

            # Write a text with the vehicle label, the confidence score and the ID
            cv2.putText(frame, vehicle_label, (round(res.bboxes[i][0]), round(res.bboxes[i][1] - 5)),
                        font, 1, color, thickness, line_type)
        
        frame_display = frame
        cv2.imshow("Result", frame_display)
        log.debug("Frame displayed.")
        
        pbar.update(1)
        counter += 1

        # Read the new frame before starting a new iteration
        read_time_start = default_timer()
        is_reading, frame = cap.read()
        read_time += default_timer() - read_time_start

    pbar.close()

    log.info("Average FPS: {}".format(str(counter / (default_timer() - before_loop))))
    log.info("Average FPS without read time: {}".format(str(counter / (default_timer() - before_loop - read_time))))

    log.info("Total detection time: {}".format(tot_det_time))
    log.info("Total tracking time: {}".format(tot_track_time))

    log.info("Average detection time: {}".format(tot_det_time / counter))
    log.info("Average tracking time: {}".format(tot_track_time / counter))

    cv2.destroyAllWindows()
