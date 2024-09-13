from google.cloud import vision
import cv2
import json
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

    # Extract the full detected text
    all_text = texts[0].description if texts else ""

    # Filter for ASCII alphanumeric characters
    filtered_text = ''.join(char for char in all_text if char.isascii() and char.isalnum())

    # Check if the filtered text complies with the Saudi license plate format
    if license_complies_format(filtered_text):
        # Format the license plate according to the Saudi format (4 numbers-3 letters)
        formatted_plate = format_license(filtered_text)
        confidence = texts[0].score if hasattr(texts[0], 'score') else 1.0
        return formatted_plate, confidence

    return None, None

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
