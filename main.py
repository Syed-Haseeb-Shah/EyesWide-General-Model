import cv2
import numpy as np
import pyttsx3
import time

# Initialize the TTS engine
engine = pyttsx3.init()

# Threshold to detect object
thres = 0.45
# NMS threshold
nms_threshold = 0.2

# Open webcam
cap = cv2.VideoCapture(0)

# Read class names from file
classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Load the pre-trained model
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Frame buffering parameters
buffer_size = 10
frame_buffer = []

# Main loop
while True:
    # Read frame from webcam
    success, img = cap.read()
    if not success:
        break

    # Detect objects in the frame
    classIds, confs, bbox = net.detect(img, confThreshold=thres)

    # Convert bounding box and confidence values to lists
    bbox = list(bbox)
    confs = list(np.array(confs).reshape(1, -1)[0])
    confs = list(map(float, confs))

    # Apply Non-Maximum Suppression
    indices = cv2.dnn.NMSBoxes(bbox, confs, thres, nms_threshold)

    # Get class names for detected objects and speak them
    detected_classes = []
    if len(indices) > 0:
        for i in indices.flatten():
            # Extract class IDs
            classId = int(classIds[i]) - 1
            if 0 <= classId < len(classNames):
                className = classNames[classId].upper()  # Get the class name
                detected_classes.append(className)

    # Speak the detected classes
    if detected_classes:
        text_to_speak = ', '.join(detected_classes)
        engine.say(text_to_speak)  # Speak the detected classes
        engine.runAndWait()  # Wait for the speech to finish

    # Store frame in buffer
    frame_buffer.append(img)

    # If buffer is full, remove the oldest frame
    if len(frame_buffer) > buffer_size:
        frame_buffer.pop(0)

    # Display the most recent frame from the buffer
    if len(frame_buffer) > 0:
        cv2.imshow("Output", frame_buffer[-1])
#'q' key for the termination of the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
