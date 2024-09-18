import cv2
import csv
import os
import numpy as np
from ultralytics import YOLO
from Helper_util import preprocess_frame, assign_car

# Load YOLO models for vehicles and license plates
vehicle_detector = YOLO('/content/drive/MyDrive/ANPR/models/yolov8n.pt')

# Load YOLOv4 model for license plate detection
net = cv2.dnn.readNet("/content/drive/MyDrive/ANPR/models/yolov4_license_plate.weights",
                      "/content/drive/MyDrive/ANPR/models/yolov4_license_plate.cfg")

# Define video capture
cap = cv2.VideoCapture("/content/drive/MyDrive/ANPR/input/v10044g50000cgaepc3c77uf8m066png.mp4")

# Create directories to save license plate images
os.makedirs('/content/drive/MyDrive/ANPR/License_plate_images', exist_ok=True)
os.makedirs('/content/drive/MyDrive/ANPR/Processed_License_plate_images', exist_ok=True)

# Initialize CSV file for logging results
csv_file = open('/content/drive/MyDrive/ANPR/output/license_plate_detections.csv', mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Frame', 'Vehicle_ID', 'BBox_Vehicle', 'BBox_License_Plate', 'Processed_Image_Path'])

# Set vehicle class IDs
vehicle_classes = [2, 3, 5, 7]  # IDs for vehicles

def detect_license_plates(frame, net):
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    boxes = []
    confidences = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w/2)
                y = int(center_y - h/2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    return [boxes[i] + [confidences[i]] for i in indices.flatten()]

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect vehicles in the current frame
    vehicle_detections = vehicle_detector.track(frame, persist=True, tracker="bytetrack.yaml")[0]

    vehicles_detected = []
    for detection in vehicle_detections.boxes.data.tolist():
        if len(detection) == 6:
            x1, y1, x2, y2, score, class_id = detection
            track_id = None
        elif len(detection) == 7:
            x1, y1, x2, y2, track_id, score, class_id = detection
        else:
            continue

        if int(class_id) in vehicle_classes:
            vehicles_detected.append([x1, y1, x2, y2, track_id, score])

    # Detect license plates in the current frame using YOLOv4
    license_plates = detect_license_plates(frame, net)

    for license_plate in license_plates:
        x, y, w, h, score_lp = license_plate
        x1_lp, y1_lp, x2_lp, y2_lp = x, y, x + w, y + h

        # Assign license plate to a detected vehicle
        assigned_car = assign_car([x1_lp, y1_lp, x2_lp, y2_lp, score_lp, 0], vehicles_detected)
        if assigned_car != -1:
            x1_v, y1_v, x2_v, y2_v, car_id = assigned_car

            # Save the original license plate image
            license_plate_image = frame[int(y1_lp):int(y2_lp), int(x1_lp):int(x2_lp)]
            license_plate_image_path = f'/content/drive/MyDrive/ANPR/License_plate_images/frame_{frame_count}_vehicle_{car_id}.jpg'
            cv2.imwrite(license_plate_image_path, license_plate_image)

            # Process the license plate image
            processed_image = preprocess_frame(frame, x1_lp, y1_lp, x2_lp, y2_lp)
            processed_image_path = f'/content/drive/MyDrive/ANPR/Processed_License_plate_images/frame_{frame_count}_vehicle_{car_id}_processed.jpg'
            cv2.imwrite(processed_image_path, processed_image)

            # Log the detection in the CSV file
            csv_writer.writerow([frame_count, car_id, f'[{x1_v}, {y1_v}, {x2_v}, {y2_v}]',
                                 f'[{x1_lp}, {y1_lp}, {x2_lp}, {y2_lp}]', processed_image_path])

    frame_count += 1

# Release resources
cap.release()
csv_file.close()

# Apply annotations on a video:
# Open the CSV file with saved detection results
csv_file = '/content/drive/MyDrive/ANPR/output/license_plate_detections.csv'

# Open the original video
cap = cv2.VideoCapture("/content/drive/MyDrive/ANPR/input/v10044g50000cgaepc3c77uf8m066png.mp4")

# Prepare video writer to save the output video with annotations
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('/content/drive/MyDrive/ANPR/output/Saudi_annot_test.mp4', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

# Read the CSV file with detection data
with open(csv_file, 'r') as file:
    csv_reader = csv.reader(file)
    next(csv_reader)  # Skip the header

    # Initialize frame counter
    frame_count = 0
    current_frame_data = []

    for row in csv_reader:
        frame_number, vehicle_id, bbox_vehicle, bbox_license_plate, _ = row

        # Convert the string bbox into a list of integers
        bbox_vehicle = list(map(float, bbox_vehicle.strip('[]').split(',')))
        bbox_license_plate = list(map(float, bbox_license_plate.strip('[]').split(',')))

        # If the current row belongs to a new frame, process the previous frame
        if int(frame_number) != frame_count:
            # Process and annotate the previous frame
            if current_frame_data:
                ret, frame = cap.read()
                if not ret:
                    break

                for vehicle_data in current_frame_data:
                    vehicle_bbox, license_plate_bbox, vehicle_id = vehicle_data

                    # Draw bounding box for the vehicle
                    cv2.rectangle(frame, (int(vehicle_bbox[0]), int(vehicle_bbox[1])), (int(vehicle_bbox[2]), int(vehicle_bbox[3])),
                                  (255, 0, 0), 2)
                    cv2.putText(frame, f'Vehicle {vehicle_id}', (int(vehicle_bbox[0]), int(vehicle_bbox[1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                    # Draw bounding box for the license plate
                    cv2.rectangle(frame, (int(license_plate_bbox[0]), int(license_plate_bbox[1])),
                                  (int(license_plate_bbox[2]), int(license_plate_bbox[3])), (0, 255, 0), 2)
                    cv2.putText(frame, f'Plate {vehicle_id}', (int(license_plate_bbox[0]), int(license_plate_bbox[1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Write the frame to the output video
                out.write(frame)

            # Move to the next frame
            current_frame_data = []
            frame_count = int(frame_number)

        # Append vehicle and license plate bounding boxes for the current frame
        current_frame_data.append([bbox_vehicle, bbox_license_plate, vehicle_id])

# Release video resources
cap.release()
out.release()
