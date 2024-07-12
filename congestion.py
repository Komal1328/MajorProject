import cv2
from vehicle_detector import VehicleDetector
import os
import pyfirmata2
import time
from database_operations import create_connection, create_table, insert_image, retrieve_recent_image

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
inter_green_delay = 3  # Inter-green delay between signals in seconds

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

# Set up the save directory
save_directory = r"C:\Users\ASUS\Desktop\source code\Output\traffic_images"
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# Set up the video capture
cap = cv2.VideoCapture('rtsp://192.168.148.106:8554/mjpeg/1')
if not cap.isOpened():
    print("Error: Unable to open RTSP stream")
    exit()

# Traffic signal logic
v = 10  # vehicles speed 10m/s
L = 5  # average length of vehicles
G = 2  # inter-vehicle gap

initial_green_delay = 10
next_capture_time = time.time() + initial_green_delay - 5
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
