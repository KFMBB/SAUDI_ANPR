import cv2
import re
import easyocr
import langid
import pytesseract
import paddle
from paddleocr import PaddleOCR

# Initialize OCR readers
reader = easyocr.Reader(['en'])  # EasyOCR for English text
ocr = PaddleOCR(lang='ar')  # PaddleOCR for Arabic text

# Model paths
model_path = "/content/SAUDI_ANPR/models/arabic_PP-OCRv3_rec_infer/inference.pdmodel"

# Load PaddleOCR model
model = paddle.jit.load(model_path)  # Load only the model without params_path

# Mapping dictionaries to correct OCR character errors
dict_char_to_int = {'O': '0', 'I': '1', 'Z': '2', 'B': '8', 'S': '5'}
dict_int_to_char = {'0': 'O', '1': 'I', '2': 'Z', '8': 'B', '5': 'S'}

def write_csv(results, output_path):
    """
    Write the result to a CSV file.

    Args:
        results (dict): Dictionary containing the result.
        output_path (str): Path to the output CSV file.
    """
    with open(output_path, 'w') as f:
        f.write('{},{},{},{},{},{},{}\n'.format('frame_c', 'car_id', 'car_bbox',
                                                'license_plate_bbox', 'license_plate_bbox_score', 'license_number',
                                                'license_number_score'))

        for frame_nmr, cars in results.items():
            for car_id, car_data in cars.items():
                if 'car' in car_data and 'license_plate' in car_data and 'text' in car_data['license_plate']:
                    car_bbox = car_data['car']['bbox']
                    lp_bbox = car_data['license_plate']['bbox']
                    lp_text = car_data['license_plate']['text']
                    f.write('{},{},{},{},{},{},{}\n'.format(
                        frame_nmr, car_id,
                        f"[{car_bbox[0]} {car_bbox[1]} {car_bbox[2]} {car_bbox[3]}]",
                        f"[{lp_bbox[0]} {lp_bbox[1]} {lp_bbox[2]} {lp_bbox[3]}]",
                        car_data['license_plate']['bbox_score'],
                        lp_text,
                        car_data['license_plate']['text_score']
                    ))

def license_complies_format(text):
    """
    Check if the license plate text complies with the Saudi license plate format (9999-XXX).

    Args:
        text (str): License plate text.

    Returns:
        bool: True if the license plate complies with the format, False otherwise.
    """
    text = text.replace('-', '')
    return len(text) == 7 and text[:4].isdigit() and text[4:].isalpha() and text[4:].isupper()

def format_license(text):
    """
    Format the license plate text by converting characters using the mapping dictionaries.

    Args:
        text (str): License plate text.

    Returns:
        str: Formatted license plate text.
    """
    license_plate_ = ''
    mapping = {0: dict_int_to_char, 1: dict_int_to_char, 2: dict_int_to_char, 3: dict_int_to_char,
               4: dict_char_to_int, 5: dict_char_to_int, 6: dict_char_to_int}

    for j in range(7):
        license_plate_ += mapping[j].get(text[j], text[j])

    return f"{license_plate_[:4]}-{license_plate_[4:]}"

def filter_english_text(text):
    """
    Use langid to filter out non-English parts of the text.

    Args:
        text (str): The detected text from the license plate.

    Returns:
        str: Filtered English text.
    """
    text_parts = text.split()
    return ' '.join(part for part in text_parts if langid.classify(part)[0] == 'en')

def read_license_plate_combined(license_plate_crop):
    """
    Try reading license plate text with multiple OCR engines (EasyOCR and Tesseract).
    """
    detections = reader.readtext(license_plate_crop)

    if detections:
        for _, text, score in detections:
            text = text.upper().replace(' ', '')
            english_text = filter_english_text(text)
            if license_complies_format(english_text):
                return format_license(english_text), score

    text = pytesseract.image_to_string(license_plate_crop).upper().replace(' ', '')
    english_text = filter_english_text(text)
    if license_complies_format(english_text):
        return format_license(english_text), None

    return None, None

def read_license_plate_Eocr(license_plate_crop):
    """
    Read the license plate text from the given cropped threshed image and filter out non-English text.

    Args:
        license_plate_crop (Image): Cropped image containing the license plate.

    Returns:
        tuple: Tuple containing the formatted license plate text and its confidence score.
    """
    detections = reader.readtext(license_plate_crop)
    for _, text, score in detections:
        text = text.upper().replace(' ', '')
        english_text = filter_english_text(text)
        if license_complies_format(english_text):
            return format_license(english_text), score
    return None, None

def assign_car(license_plate, vehicle_track_ids, tolerance=0.1):
    """
    Retrieve the vehicle coordinates and ID based on the license plate coordinates with a margin of tolerance.

    Args:
        license_plate (tuple): Tuple containing the coordinates of the license plate (x1, y1, x2, y2, score, class_id).
        vehicle_track_ids (list): List of vehicle track IDs and their corresponding coordinates.
        tolerance (float): Percentage of tolerance to make the bounding box check more lenient. Default is 10%.

    Returns:
        tuple: Tuple containing the vehicle coordinates (x1, y1, x2, y2) and ID.
    """
    x1_lp, y1_lp, x2_lp, y2_lp, _, _ = license_plate

    for vehicle in vehicle_track_ids:
        x1_v, y1_v, x2_v, y2_v, track_id, _ = vehicle
        width_v, height_v = x2_v - x1_v, y2_v - y1_v
        tol_w, tol_h = tolerance * width_v, tolerance * height_v

        if (x1_lp > x1_v - tol_w and y1_lp > y1_v - tol_h and
                x2_lp < x2_v + tol_w and y2_lp < y2_v + tol_h):
            return x1_v, y1_v, x2_v, y2_v, track_id
    return -1, -1, -1, -1, -1

def preprocess_frame(frame, x1, y1, x2, y2):
    """
    Preprocess the given frame to crop the license plate, convert it to grayscale,
    and apply adaptive thresholding.

    Args:
        frame (numpy array): The input frame containing the car/license plate.
        x1, y1, x2, y2 (int): Bounding box coordinates of the license plate.

    Returns:
        license_plate_crop_thresh (numpy array): Preprocessed license plate image after adaptive thresholding.
    """
    license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2)]
    license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)

    return cv2.adaptiveThreshold(
        license_plate_crop_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

def format_license_re(license_text):
    """
    Use regex to format license text according to Saudi license plate standards.

    Args:
        license_text (str): Raw license text.

    Returns:
        str: Formatted license text.
    """
    match = re.match(r"(\d{1,4})([A-Za-z]{1,3})", license_text)
    return f"{match.group(1)}-{match.group(2)}".upper() if match else license_text.upper()

def validate_license_plate(license_text):
    """
    Validate and correct the license plate text based on Saudi standards.

    Args:
        license_text (str): License plate text.

    Returns:
        str: Validated license text.
    """
    allowed_letters = 'ABCDEFGHJKLMNPQRSTUVWXYZ'
    allowed_digits = '0123456789'

    valid_license = ''.join(
        char if char in allowed_digits or char in allowed_letters else '_'
        for char in license_text
    )
    return valid_license

