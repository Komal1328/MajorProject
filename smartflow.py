import cv2
from vehicle_detector import VehicleDetector
import os
import pyfirmata2
import time
from database_operations import create_connection, create_table, insert_image, retrieve_recent_image
import numpy as np
import pytesseract
from imutils.object_detection import non_max_suppression

# Try connecting to Arduino
try:
    board = pyfirmata2.Arduino('COM4')
except Exception as e:
    print(f"Error: Unable to connect to Arduino on COM4. {e}")
    exit()

# Define the pins for the traffic signals for four lanes within the available range
pins = {
    'signal1': {'red': 13, 'yellow': 12, 'green': 11},
    'signal2': {'red': 10, 'yellow': 9, 'green': 8},
    'signal3': {'red': 7, 'yellow': 6, 'green': 5},
    'signal4': {'red': 4, 'yellow': 3, 'green': 2}
}

# Delays in seconds
red_delay = 10  # 10 seconds
yellow_delay = 2  # 2 seconds
inter_green_delay = 1  # Inter-green delay between signals in seconds


# Function to set the light state (HIGH or LOW) for a specific signal and color
def set_light(signal, color, state):
    pin = pins[signal][color]
    board.digital[pin].write(state)


# Function to handle a traffic light cycle for a specific signal
def traffic_light_cycle(signal, green_delay):
    print(f"{signal} cycle started with green delay: {green_delay}")

    # Set all other signals' red lights
    for key in pins:
        if key != signal:
            set_light(key, 'red', 1)

    # Turn on the green light
    set_light(signal, 'red', 0)
    set_light(signal, 'yellow', 0)
    set_light(signal, 'green', 1)
    time.sleep(green_delay)

    # Turn on the yellow light
    set_light(signal, 'green', 0)
    set_light(signal, 'yellow', 1)
    time.sleep(yellow_delay)

    # Turn on the red light
    set_light(signal, 'yellow', 0)
    set_light(signal, 'red', 1)
    time.sleep(inter_green_delay)


# Function to handle emergencies by giving priority
def emergency(img_path):
    def decode_predictions(scores, geometry):
        (numRows, numCols) = scores.shape[2:4]
        rects = []
        confidences = []

        for y in range(0, numRows):
            scoresData = scores[0, 0, y]
            xData0 = geometry[0, 0, y]
            xData1 = geometry[0, 1, y]
            xData2 = geometry[0, 2, y]
            xData3 = geometry[0, 3, y]
            anglesData = geometry[0, 4, y]

            for x in range(0, numCols):
                if scoresData[x] < 0.5:
                    continue

                (offsetX, offsetY) = (x * 4.0, y * 4.0)

                angle = anglesData[x]
                cos = np.cos(angle)
                sin = np.sin(angle)

                h = xData0[x] + xData2[x]
                w = xData1[x] + xData3[x]

                endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
                endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
                startX = int(endX - w)
                startY = int(endY - h)

                rects.append((startX, startY, endX, endY))
                confidences.append(scoresData[x])

        return (rects, confidences)

    east_model_path = 'frozen_east_text_detection.pb'
    net = cv2.dnn.readNet(east_model_path)

    min_confidence = 0.5
    width = 320
    height = 320
    padding = 0.05

    image = cv2.imread(img_path)
    orig = image.copy()
    (origH, origW) = image.shape[:2]

    (newW, newH) = (width, height)
    rW = origW / float(newW)
    rH = origH / float(newH)

    image = cv2.resize(image, (newW, newH))
    (H, W) = image.shape[:2]

    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
                                 (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    (scores, geometry) = net.forward(["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"])

    (rects, confidences) = decode_predictions(scores, geometry)
    boxes = non_max_suppression(np.array(rects), probs=confidences)

    results = []

    for (startX, startY, endX, endY) in boxes:
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)

        dX = int((endX - startX) * padding)
        dY = int((endY - startY) * padding)

        startX = max(0, startX - dX)
        startY = max(0, startY - dY)
        endX = min(origW, endX + (dX * 2))
        endY = min(origH, endY + (dY * 2))

        roi = orig[startY:endY, startX:endX]

        config = ("-l eng --oem 1 --psm 7")
        text = pytesseract.image_to_string(roi, config=config)

        results.append(((startX, startY, endX, endY), text))

    results = sorted(results, key=lambda r: r[0][1])

    for ((startX, startY, endX, endY), text) in results:
        text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
        output = orig.copy()
        cv2.rectangle(output, (startX, startY), (endX, endY),
                      (0, 0, 255), 2)
        cv2.putText(output, text, (startX, startY - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        if "AMBULANCE" in text:
            output_image_path = r"C:\Users\ASUS\Desktop\source code\Output\output_images\output.jpg"

            # Save the output image
            cv2.imwrite(output_image_path, output)

            print("AMBULANCE detected.")

            return 1

    return 0


# Detect number of vehicles
def congestion(img_path):
    print("Processing image for vehicle detection...")
    vd = VehicleDetector()

    output_folder = r"C:\Users\ASUS\Desktop\source code\Output\output_images"
    img = cv2.imread(img_path)

    if img is None or img.size == 0:
        print("Error: Unable to read the image or image has invalid dimensions.")
        return 0

    vehicle_boxes = vd.detect_vehicles(img)
    vehicle_count = len(vehicle_boxes)

    for box in vehicle_boxes:
        x, y, w, h = box
        cv2.rectangle(img, (x, y), (x + w, y + h), (25, 0, 180), 3)
        cv2.putText(img, "Vehicles: " + str(vehicle_count), (20, 50), 0, 2, (100, 200, 0), 3)

    img_name = os.path.basename(img_path)
    output_path = os.path.join(output_folder, img_name)
    cv2.imwrite(output_path, img)

    cv2.imshow("Vehicles", img)
    cv2.waitKey(1)
    print("Total number of vehicles in congestion:", vehicle_count)
    return vehicle_count


def get_image_path(signal_number):
    return f"C:\\Users\\ASUS\\Desktop\\source code\\images\\{signal_number}.jpg"

amb_image_path = r"C:\Users\ASUS\Desktop\source code\images\6.jpg"

# Set up the save directory
save_directory = r"C:\Users\ASUS\Desktop\source code\Output\traffic_images"
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# Set up the video capture
cap = cv2.VideoCapture('rtsp://192.168.129.106:8554/mjpeg/1')
if not cap.isOpened():
    print("Error: Unable to open RTSP stream")
    exit()

# Traffic signal logic
v = 10  # vehicles speed 10m/s
L = 5  # average length of vehicles
G = 2  # inter-vehicle gap

initial_green_delay = 10
next_capture_time = time.time() + initial_green_delay - 5
emergency_check_interval = 30  # Check for emergency every 30 seconds
next_emergency_check = time.time() + emergency_check_interval

count = 0
signal_order = ['signal1', 'signal2', 'signal3', 'signal4']
signal_index = 0

connection = create_connection()
if connection:
    create_table(connection)
    print("Database connected")
else:
    print("Error: Unable to connect to the database")
    exit()

# Main loop to run the traffic light cycles for all four signals
while True:
    try:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame from the RTSP stream.")
            break

        cv2.imshow("Live Transmission", frame)

        current_time = time.time()

        # Check for emergency every 40 seconds
        if current_time >= next_emergency_check:
            recent_image_path = retrieve_recent_image(connection, save_directory)
            if recent_image_path:
                # if emergency(recent_image_path) == 1:
                if emergency(amb_image_path) == 1:
                    print("Ambulance detected!")
                    traffic_light_cycle('signal1', 5)  # Set green delay of 5 seconds for signal1
                else:
                    print("No ambulance")
            next_emergency_check = current_time + emergency_check_interval

        if current_time >= next_capture_time:
            count += 1
            filename = f"{count}.png"
            cv2.imwrite(filename, frame)
            print("Image saved as:", filename)

            test_images_folder = r"C:\Users\ASUS\Desktop\source code\Output\retrieve_images"
            if not os.path.exists(test_images_folder):
                os.makedirs(test_images_folder)
            test_image_path = os.path.join(test_images_folder, filename)
            if os.path.exists(test_image_path):
                os.remove(test_image_path)
            os.rename(filename, test_image_path)
            print("Image saved in 'test_images' folder.")

            if os.path.exists(test_image_path):
                print("File exists. Uploading to database...")
                insert_image(connection, test_image_path)
            else:
                print("Error: File not found. Unable to upload to database.")

            recent_image_path = retrieve_recent_image(connection, save_directory)
            if recent_image_path:
                current_signal = signal_order[signal_index]
                # vehicle_count = congestion(recent_image_path)  # Use recent image path
                img_path = get_image_path(signal_index + 1)
                vehicle_count = congestion(img_path)

                green_delay = (L + (vehicle_count - 1) * (L + G)) / v
                print("Green Delay:", green_delay)
            else:
                green_delay = initial_green_delay

            next_capture_time = current_time + green_delay - 5

            current_signal = signal_order[signal_index]
            traffic_light_cycle(current_signal, green_delay)
            signal_index = (signal_index + 1) % len(signal_order)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        print(f"Error: {e}")
        break

cap.release()
cv2.destroyAllWindows()
