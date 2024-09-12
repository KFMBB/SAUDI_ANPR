from ultralytics import YOLO
import cv2
import easyocr
from Helper_util import assign_car, write_csv,annotate_image

# The steps of the project:
# 1- We'll use YOLO for license plate detection1 (object detection1).
# 2- We'll use SORT model to keep tracking our object on the road (object tracking).
# 3- Lastly using easy OCR we'll read the license plate contents.
reader = easyocr.Reader(['ar', 'en'])
results = {}  # We'll save our result for processing later.
# load models
# We're using two pretrained models:
# 1- The first model will focus on detecting cars while the other will focus on detecting license plates:
model = YOLO('models/yolov8n.pt')  # Load the nano version of the yolo model.
# 2- The second model will be used to detect the license plates.
license_plate_detector = YOLO('models/best_license_plate_detector.pt')

# Since we're detecting vehicles we'll set a list with the YOLO class_id for vehicles.
# why? well to make our model less computationally expensive and make it focus on the task at hand
vehicles = [2, 3, 5, 7]
# read frames
frame_c = -1  # We'll assign the frame counter as a unique key for our result.
threshold = 64  # Variable we'll use when using cv2 thresholding.
vid_img = int(input('Hello, welcome to the test case, please select from the following: [Video: 0, Image: 1]'))
if vid_img == 0:
    vid_path = input('Please enter a video file directory: ')
    cap = cv2.VideoCapture(vid_path)  # Load user video.
    isReadingFrames = True  # This is a boolean flag that keeps track of whether the frame was read or not.
    while isReadingFrames:
        frame_c += 1
        isReadingFrames, frame = cap.read()
        if isReadingFrames:
            results[frame_c] = {}
            # Now apply each model to the current frame:
            # vehicle detection1 and tracking:
            detections = model.track(frame, persist=True, tracker="bytetrack.yaml")[0]  # the [0] will unpack the models
            # bbox, confidence and class_id as a tuple.
            vehicles_detected2 = []  # Store each vehicle detected in the list.
            for detection1 in detections.boxes.data.tolist():
                x1, y1, x2, y2, track_id, score, class_id = detection1
                if int(class_id) in vehicles:  # If the object detected is a vehicle:
                    vehicles_detected2.append([x1, y1, x2, y2, track_id, score])  # Add bbox, confidence to the list of
                    # detected vehicles.

            # detect the license plates:
            license_plates = license_plate_detector(frame)[0]  # same as the vehicles detections.
            for license_plate in license_plates.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = license_plate

                # Now that we have detected cars and license plates, we'll assign license plates to cars:
                x1_v, y1_v, x2_v, y2_v, car_id = assign_car(license_plate, vehicles_detected2)

                if car_id != -1:  # If the car was assigned to a license plate successfully go to the processing pipeline

                    # crop license plate:
                    license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]
                    # we essentially took the bbox cords and then processed the whole frame, we took slices from y1 till
                    # y2 and then x1 till x2 and lastly the rgb or bgr.

                    # process license plate number: why? well to make it easier for the ocr model to process, we'll grey
                    # scale the license plate as colors don't really help the model with reading a plate.
                    license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                    # apply threshold: If the pixel value is less than threshold it becomes white (255),
                    # else it becomes black that's why we call it binary thresholding.
                    not_needed, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, threshold, 255,
                                                                          cv2.THRESH_BINARY_INV)

                    # Now that we have processed our license plate image we'll read it using easy ocr:
                    ocr_results = reader.readtext(license_plate_crop_gray)
                    license_plate_text = ''.join([result[1].upper().replace(' ', '') for result in ocr_results])
                    license_plate_text_conf = [result[2] for result in ocr_results]
                    # Save the result:
                    if license_plate_text is not None:
                        results[frame_c][car_id] = {'car': {'bbox': [x1_v, y1_v, x2_v, y2_v]},
                                                    'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                      'text': license_plate_text,
                                                                      'bbox_score': score,
                                                                      'text_score': license_plate_text_conf}}

    # write result to user csv file
    results_path = input('Please input a result file path (.csv)')
    write_csv(results, results_path)
else:
    # Handling image input
    image_path = input('Please enter an image file directory: ')
    frame = cv2.imread(image_path)  # Load image
    if frame is not None:
        # Initialize result for a single image
        results[0] = {}

        # Detect vehicles
        detection1 =  model.track(frame, persist=True, tracker="bytetrack.yaml")[0]
        vehicles_detected2 = []
        for detection1 in detection1.boxes.data.tolist():
            x1, y1, x2, y2, track_id, score, class_id = detection1
            if int(class_id) in vehicles:
                vehicles_detected2.append([x1, y1, x2, y2, track_id, score])

        # Detect license plates
        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # Assign license plate to a vehicle
            x1_v, y1_v, x2_v, y2_v, car_id = assign_car(license_plate, vehicles_detected2)

            if car_id != -1:
                # Crop and preprocess license plate image
                license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, threshold, 255,
                                                             cv2.THRESH_BINARY_INV)

                # OCR to extract text from license plate
                ocr_results = reader.readtext(license_plate_crop_thresh)
                license_plate_text = ''.join([result[1].upper().replace(' ', '') for result in ocr_results])
                license_plate_text_conf = [result[2] for result in ocr_results]

                # Save result
                if license_plate_text:
                    results[0][car_id] = {
                        'car': {'bbox': [x1_v, y1_v, x2_v, y2_v]},
                        'license_plate': {
                            'bbox': [x1, y1, x2, y2],
                            'text': license_plate_text,
                            'bbox_score': score,
                            'text_score': license_plate_text_conf
                        }
                    }

        # Annotate and save the image
        annotated_image_path = input('Please input a path to save the annotated image: ')
        annotated_image = annotate_image(frame, results[0])  # Function to draw bounding boxes and text
        cv2.imwrite(annotated_image_path, annotated_image)

    else:
        print('Error: Unable to load image. Please check the file path.')