from ultralytics import YOLO
import cv2
# The steps of the project:
# 1- We'll use YOLO for license plate detection (object detection).
# 2- We'll use Bytetracker model to keep tracking our object on the road (object tracking).
# 3- Lastly using google vision api we'll read the license plate contents.
import Helper_util
from Helper_util import assign_car,write_csv, preprocess_frame,read_license_plate_Eocr,read_license_plate_combined

results = {}  # We'll save our results for processing later.

# load models
# We're using two pre-trained models:
# 1- The first model will focus on detecting cars while the other will focus on detecting license plates:
model = YOLO('models\yolov8m.pt')  # Load the nano version of the yolo model.
# 2- The second model will be used to detect the license plates.
license_plate_detector = YOLO('models\best_license_plate_detector.pt')
# load video
cap = cv2.VideoCapture("input/testing.mp4")
# Since we're detecting vehicles we'll set a list with the YOLO class_id for vehicles.
# why? well to make our model less computationally expensive and make it focus on the task at hand
vehicles = [2, 3, 5, 7]

# read frames
frame_c = -1  # We'll assign the frame counter as a unique key for our results.
threshold = 64  # Variable we'll use when using cv2 thresholding.
isReadingFrames = True  # This is a boolean flag that keeps track of whether the frame was read or not.
while isReadingFrames:
    frame_c += 1
    isReadingFrames, frame = cap.read()
    if isReadingFrames:
        results[frame_c] = {}
        # Now apply each model to the current frame:
        # Vehicle detection and tracking:
        detections = model.track(frame, persist = True, tracker="bytetrack.yaml")[0]  # the [0] will unpack the models bbox, confidence and class_id as a tuple.
        vehicles_detected = []  # Store each vehicle detected in the list.
        for detection in detections.boxes.data.tolist():
            if len(detection) == 6:  # Format: [x1, y1, x2, y2, score, class_id]
                x1, y1, x2, y2, score, class_id = detection
                track_id = None
            elif len(detection) == 7:  # Format: [x1, y1, x2, y2, track_id, score, class_id]
                x1, y1, x2, y2, track_id, score, class_id = detection
            else:
                print(f"Unexpected detection format: {detection}")
                continue  # Skip this detection or handle it as needed
                
            if int(class_id) in vehicles:  # If the object detected is a vehicle:
                vehicles_detected.append([x1, y1, x2, y2, track_id, score])  # Add bbox, confidence to the list of detected vehicles.

        # detect the license plates:
        license_plates = license_plate_detector(frame)[0]  # same as the vehicles detections.
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # Now that we have detected cars and license plates, we'll assign license plates to cars:
            x1_v, y1_v, x2_v, y2_v, car_id = assign_car(license_plate, vehicles_detected)

            if car_id != -1:  # If the car was assigned to a license plate successfully go to the processing pipeline

                license_plate_crop_thresh = preprocess_frame(frame, x1, y1, x2, y2)

                # Now that we have processed our license plate image we'll read it using easy ocr:
                license_plate_text, license_plate_text_score = read_license_plate_combined(license_plate_crop_thresh)

                # Save the results:
                if license_plate_text is not None:
                    results[frame_c][car_id] = {'car': {'bbox': [x1_v, y1_v, x2_v, y2_v]},
                                                'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                  'text': license_plate_text,
                                                                  'bbox_score': score,
                                                                  'text_score': license_plate_text_score}}

# write results
write_csv(results, 'output/results.csv')
