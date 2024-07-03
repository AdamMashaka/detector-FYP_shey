import os
from ultralytics import YOLO
import torch
import cv2
import numpy as np
from util import get_car, read_license_plate, write_csv, detect_phone_usage
from sort.sort import Sort

results = {}
mot_tracker = Sort()

# Load models
coco_model = YOLO('yolov9c.pt')
license_plate_detector = YOLO('./models/license_plate_detector.pt')
phone_detector = YOLO('./models/phone_detector.pt')  # Load a model trained to detect phones

# Load camera
cap = cv2.VideoCapture(0)  # 0 is typically the default camera index for the laptop camera

vehicles = [2, 3, 5, 7]  # Vehicle class IDs

frame_nmr = -1
ret = True

while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if ret:
        results[frame_nmr] = {}

        # Detect vehicles
        detections = coco_model(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])

        # Track vehicles
        track_ids = mot_tracker.update(np.asarray(detections_))

        # Detect license plates
        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # Assign license plate to car
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

            if car_id != -1:
                # Crop license plate
                license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]

                # Process license plate
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop)

                if license_plate_text is not None:
                    results[frame_nmr][car_id] = {
                        'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                        'license_plate': {
                            'bbox': [x1, y1, x2, y2],
                            'text': license_plate_text,
                            'bbox_score': score,
                            'text_score': license_plate_text_score
                        }
                    }

        # Detect phone usage
        phone_usages = {}
        for x1, y1, x2, y2, car_id in track_ids:
            # Crop the driver region (modify this region based on your video)
            driver_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
            phone_usages[car_id] = detect_phone_usage(driver_crop)

        for car_id, is_using_phone in phone_usages.items():
            if car_id in results[frame_nmr]:
                results[frame_nmr][car_id]['phone_usage'] = is_using_phone

    # Display the frame
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Write results
write_csv(results, './tests.csv')

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
