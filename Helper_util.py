from google.cloud import vision
import cv2
import json
import re
json_path = "API_Credentials/client_secret_616972654566-tbjlta6vijmal5rlq07hoba3nja2f7n3.apps.googleusercontent.com.json"

with open(json_path, 'r') as file:
    config = json.load(file)

client = vision.ImageAnnotatorClient.from_service_account_json(json_path)
# Mapping dictionaries for character conversion
# Mapping characters that can be confused with numbers
dict_char_to_int = {
    'O': '0',  # O looks like 0
    'I': '1',  # I looks like 1
    'Z': '2',  # Z looks like 2
    'B': '8',  # B looks like 8
    'S': '5'   # S looks like 5
}

# Mapping numbers that can be confused with letters
dict_int_to_char = {
    '0': 'O',  # 0 can look like O
    '1': 'I',  # 1 can look like I
    '2': 'Z',  # 2 can look like Z
    '8': 'B',  # 8 can look like B
    '5': 'S'   # 5 can look like S
}

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

        for frame_nmr in results.keys():
            for car_id in results[frame_nmr].keys():
                print(results[frame_nmr][car_id])
                if 'car' in results[frame_nmr][car_id].keys() and \
                   'license_plate' in results[frame_nmr][car_id].keys() and \
                   'text' in results[frame_nmr][car_id]['license_plate'].keys():
                    f.write('{},{},{},{},{},{},{}\n'.format(frame_nmr,
                                                            car_id,
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['car']['bbox'][0],
                                                                results[frame_nmr][car_id]['car']['bbox'][1],
                                                                results[frame_nmr][car_id]['car']['bbox'][2],
                                                                results[frame_nmr][car_id]['car']['bbox'][3]),
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][0],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][1],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][2],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][3]),
                                                            results[frame_nmr][car_id]['license_plate']['bbox_score'],
                                                            results[frame_nmr][car_id]['license_plate']['text'],
                                                            results[frame_nmr][car_id]['license_plate']['text_score'])
                            )
        f.close()


def license_complies_format(text):
    """
    Check if the license plate text complies with the Saudi license plate format (9999-XXX).

    Args:
        text (str): License plate text.

    Returns:
        bool: True if the license plate complies with the format, False otherwise.
    """
    # Remove any potential dash in the license plate
    text = text.replace('-', '')

    # Check if the length is exactly 7 characters (4 digits + 3 letters)
    if len(text) != 7:
        return False

    # Check the first 4 characters are digits
    if not text[:4].isdigit():
        return False

    # Check the last 3 characters are uppercase letters
    if not text[4:].isalpha() or not text[4:].isupper():
        return False

    return True


def format_license(text):
    """
    Format the license plate text by converting characters using the mapping dictionaries.

    Args:
        text (str): License plate text.

    Returns:
        str: Formatted license plate text.
    """
    license_plate_ = ''

    # Mapping based on the position in the Saudi license plate format
    # Positions 0-3 are digits, positions 4-6 are letters
    mapping = {0: dict_int_to_char, 1: dict_int_to_char, 2: dict_int_to_char, 3: dict_int_to_char,
               4: dict_char_to_int, 5: dict_char_to_int, 6: dict_char_to_int}

    for j in range(7):
        if text[j] in mapping[j].keys():
            license_plate_ += mapping[j][text[j]]
        else:
            license_plate_ += text[j]

    # Insert a dash between the numbers and letters (e.g., 1234-ABC)
    formatted_license_plate = license_plate_[:4] + '-' + license_plate_[4:]

    return formatted_license_plate


def read_license_plate(cropped_plate):
    """
    Detect and format text from a cropped license plate using Google Vision API.

    Args:
        cropped_plate (Image): Cropped image containing the license plate.

    Returns:
        str: Formatted license plate text.
    """
    # Convert the cropped plate image to bytes
    success, encoded_image = cv2.imencode('.png', cropped_plate)
    content = encoded_image.tobytes()

    image = vision.Image(content=content)

    # Perform text detection
    response = client.text_detection(image=image)
    texts = response.text_annotations

    if not texts:
        print("No text detected in the plate.")
        return None, None

    # Extract all detected text
    all_text = texts[0].description if texts else ""

    # Filter for ASCII alphanumeric characters and spaces
    filtered_text = ''.join(char for char in all_text if char.isascii() and (char.isalnum() or char.isspace()))

    # Split the text into parts
    parts = filtered_text.split()

    # Check if "KSA" is present
    if "KSA" in parts:
        # For plates with KSA, take the last part (should be the right side)
        plate_text = parts[-1]
    else:
        # For other plates, combine all parts, focusing on sequences of 3-4 characters
        plate_text = ' '.join(part for part in parts if len(part) >= 3 and len(part) <= 4)

    # Ensure we have both letters and numbers
    if not (any(c.isalpha() for c in plate_text) and any(c.isdigit() for c in plate_text)):
        # If not, try to extract letters and numbers separately
        letters = ''.join(c for c in filtered_text if c.isalpha())
        numbers = ''.join(c for c in filtered_text if c.isdigit())
        plate_text = f"{numbers} {letters}"  # Changed order here

    # Remove KSA if it's still present in the final plate_text
    plate_text = plate_text.replace("KSA", "").strip()

    # Split the plate text into numbers and letters
    numbers = ''.join(c for c in plate_text if c.isdigit())
    letters = ''.join(c for c in plate_text if c.isalpha())

    # If the number part has more than 4 digits, remove the last digit
    if len(numbers) > 4:
        numbers = numbers[:4]

    # Recombine numbers and letters
    plate_text = f"{numbers} {letters}"  # Changed order here

    print("Detected text in plate:", plate_text.strip())

    # Check if the filtered text complies with the Saudi license plate format
    if license_complies_format(plate_text):
        # Format the license plate according to the Saudi format (4 numbers-3 letters)
        formatted_plate = format_license(plate_text)
        confidence = texts[0].score if hasattr(texts[0], 'score') else 0.0
        return formatted_plate, confidence

    return None, None



def read_plate2(cropped_plate):
    # Convert the cropped plate image to bytes
    success, encoded_image = cv2.imencode('.png', cropped_plate)
    content = encoded_image.tobytes()

    # Create a Vision API client
    client = vision.ImageAnnotatorClient()

    image = vision.Image(content=content)

    try:
        # Perform text detection
        response = client.text_detection(image=image)
        texts = response.text_annotations

        if not texts:
            print("No text detected in the plate.")
            return "", 0.0

        # Extract all detected text (first entry contains the full description)
        all_text = texts[0].description if texts else ""

        # Filter for ASCII alphanumeric characters and spaces
        filtered_text = ''.join(char for char in all_text if char.isascii() and (char.isalnum() or char.isspace()))

        # Split the text into parts
        parts = filtered_text.split()

        # Check if "KSA" is present
        if "KSA" in parts:
            # For plates with KSA, take the last part (should be the right side)
            plate_text = parts[-1]
        else:
            # For other plates, combine all parts, focusing on sequences of 3-4 characters
            plate_text = ' '.join(part for part in parts if len(part) >= 3 and len(part) <= 4)

        # Ensure we have both letters and numbers
        if not (any(c.isalpha() for c in plate_text) and any(c.isdigit() for c in plate_text)):
            # If not, try to extract letters and numbers separately
            letters = ''.join(c for c in filtered_text if c.isalpha())
            numbers = ''.join(c for c in filtered_text if c.isdigit())
            plate_text = f"{numbers} {letters}"  # Changed order here

        # Remove KSA if it's still present in the final plate_text
        plate_text = plate_text.replace("KSA", "").strip()

        # Split the plate text into numbers and letters
        numbers = ''.join(c for c in plate_text if c.isdigit())
        letters = ''.join(c for c in plate_text if c.isalpha())

        # If the number part has more than 4 digits, remove the last digit
        if len(numbers) > 4:
            numbers = numbers[:4]

        # Recombine numbers and letters
        plate_text = f"{numbers} {letters}"  # Changed order here

        # Get the confidence score from individual words or the overall annotation
        confidence_score = 0.0
        if response.full_text_annotation:
            confidence_score = sum(p.confidence for p in response.full_text_annotation.pages[0].blocks) / len(response.full_text_annotation.pages[0].blocks)
        else:
            confidence_score = sum(t.confidence for t in response.text_annotations[1:]) / len(response.text_annotations[1:])

        print("Detected text in plate:", plate_text.strip())
        print("Confidence score:", confidence_score)

        if response.error.message:
            raise Exception(f'{response.error.message}\nFor more info on error messages, check: '
                            'https://cloud.google.com/apis/design/errors')

        return plate_text.strip(), confidence_score

    except Exception as e:
        print(f"Error during text detection: {e}")
        return "", 0.0




# More loose assign_car:
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
    x1_lp, y1_lp, x2_lp, y2_lp, score_lp, class_id_lp = license_plate

    foundIt = False
    for j in range(len(vehicle_track_ids)):
        x1_v, y1_v, x2_v, y2_v, track_id, class_id_v = vehicle_track_ids[j]

        # Calculate the tolerance as a fraction of the vehicle's width and height
        width_v = x2_v - x1_v
        height_v = y2_v - y1_v
        tol_w = tolerance * width_v
        tol_h = tolerance * height_v

        # Check if the license plate is within the vehicle's bounding box with some tolerance
        if (x1_lp > (x1_v - tol_w) and y1_lp > (y1_v - tol_h) and
            x2_lp < (x2_v + tol_w) and y2_lp < (y2_v + tol_h)):
            car_indx = j
            foundIt = True  # If the car's license plate was found then break and assign it.
            break

    if foundIt:
        # Return the vehicle's bounding box and ID
        return vehicle_track_ids[car_indx][:5]
    # Return None if no matching vehicle is found
    return -1, -1, -1, -1, -1


# Preprocess frame by cropping the license plate, converting to grayscale, and applying adaptive thresholding
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
    # Crop the license plate from the frame using bounding box coordinates
    license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

    # Convert the cropped license plate to grayscale
    license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to the grayscale image
    license_plate_crop_thresh = cv2.adaptiveThreshold(
        license_plate_crop_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    return license_plate_crop_thresh

def read_plate(cropped_plate):
    # Convert the cropped plate image to bytes
    success, encoded_image = cv2.imencode('.png', cropped_plate)
    content = encoded_image.tobytes()

    # Create a Vision API client
    client = vision.ImageAnnotatorClient()

    image = vision.Image(content=content)

    try:
        # Perform text detection
        response = client.text_detection(image=image)
        texts = response.text_annotations

        if not texts:
            print("No text detected in the plate.")
            return "", 0.0

        # Extract all detected text (first entry contains the full description)
        all_text = texts[0].description if texts else ""

        # Filter for ASCII alphanumeric characters and spaces
        filtered_text = ''.join(char for char in all_text if char.isascii() and (char.isalnum() or char.isspace()))

        # Split the text into parts
        parts = filtered_text.split()

        # Check if "KSA" is present
        if "KSA" in parts:
            # For plates with KSA, take the last part (should be the right side)
            plate_text = parts[-1]
        else:
            # For other plates, combine all parts, focusing on sequences of 3-4 characters
            plate_text = ' '.join(part for part in parts if len(part) >= 3 and len(part) <= 4)

        # Ensure we have both letters and numbers
        if not (any(c.isalpha() for c in plate_text) and any(c.isdigit() for c in plate_text)):
            # If not, try to extract letters and numbers separately
            letters = ''.join(c for c in filtered_text if c.isalpha())
            numbers = ''.join(c for c in filtered_text if c.isdigit())
            plate_text = f"{numbers} {letters}"  # Changed order here

        # Remove KSA if it's still present in the final plate_text
        plate_text = plate_text.replace("KSA", "").strip()

        # Split the plate text into numbers and letters
        numbers = ''.join(c for c in plate_text if c.isdigit())
        letters = ''.join(c for c in plate_text if c.isalpha())

        # If the number part has more than 4 digits, remove the last digit
        if len(numbers) > 4:
            numbers = numbers[:4]

        # Recombine numbers and letters
        plate_text = f"{numbers} {letters}"  # Changed order here

        # Extract confidence scores similar to the `detect_document` structure
        block_confidences = []
        word_confidences = []
        symbol_confidences = []

        if response.full_text_annotation:
            for page in response.full_text_annotation.pages:
                for block in page.blocks:
                    block_confidences.append(block.confidence)

                    for paragraph in block.paragraphs:
                        for word in paragraph.words:
                            word_confidences.append(word.confidence)

                            for symbol in word.symbols:
                                symbol_confidences.append(symbol.confidence)

            # Average confidence scores
            avg_block_confidence = sum(block_confidences) / len(block_confidences) if block_confidences else 0.0
            avg_word_confidence = sum(word_confidences) / len(word_confidences) if word_confidences else 0.0
            avg_symbol_confidence = sum(symbol_confidences) / len(symbol_confidences) if symbol_confidences else 0.0

        else:
            avg_block_confidence = avg_word_confidence = avg_symbol_confidence = 0.0

        print(f"Detected text in plate: {plate_text.strip()}")
        print(f"Average block confidence: {avg_block_confidence}")
        print(f"Average word confidence: {avg_word_confidence}")
        print(f"Average symbol confidence: {avg_symbol_confidence}")

        if response.error.message:
            raise Exception(f'{response.error.message}\nFor more info on error messages, check: '
                            'https://cloud.google.com/apis/design/errors')

        # You can choose which confidence score to return; for example, word confidence
        return plate_text.strip(), avg_word_confidence

    except Exception as e:
        print(f"Error during text detection: {e}")
        return "", 0.0


# New format_license_re function using regex
def format_license_re(license_text):
    # Regex pattern to match a standard Saudi license plate pattern (modify as needed)
    # Assuming Saudi plates follow the format: 3 digits + 3 letters (Example: 123ABC)
    pattern = r"(\d{1,3})([A-Za-z]{1,3})"
    
    match = re.match(pattern, license_text)
    if match:
        digits, letters = match.groups()
        return f"{digits}-{letters}".upper()
    else:
        return license_text.upper()

# Check and replace unaccepted characters for Saudi license plates
def validate_license_plate(license_text):
    # Saudi plates generally allow A-Z (some limited set of letters, adjust accordingly)
    allowed_letters = 'ABCDEFGHJKLMNPQRSTUVWXYZ'  # Exclude I, O, Q for example
    allowed_digits = '0123456789'
    
    valid_license = []
    
    for char in license_text:
        if char in allowed_digits:
            valid_license.append(char)
        elif char in allowed_letters:
            valid_license.append(char)
        else:
            # Replace with a placeholder or closest valid character
            valid_license.append('_')  # Placeholder or handle replacement logic here
    
    return ''.join(valid_license)
