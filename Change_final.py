import cv2
import dlib
import numpy as np
import face_recognition
from scipy.spatial import distance
import time
import pygame

# Function to check if the detected face matches the driver's face
def is_driver_face(detected_face_encoding, driver_face_encoding):
    # Compare the face encodings using a threshold
    threshold = 0.6
    matches = face_recognition.compare_faces([driver_face_encoding], detected_face_encoding, tolerance=threshold)
    return any(matches)

# Load the driver's image and encode the face
driver_image = face_recognition.load_image_file("driver image.jpg")
driver_face_encoding = face_recognition.face_encodings(driver_image)[0]

# Initialize video capture and face detector
cap = cv2.VideoCapture(0)
hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

pygame.mixer.init()
pygame.mixer.music.load('C:\\Users\\HP\\Desktop\\dt2\\music.wav')  # Replace 'path/to/custom_sound.mp3' with the path to your custom sound file

counter = 0  # Initialize the counter
face_landmarks = None  # Define face_landmarks outside the loop

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = hog_face_detector(gray)

    for face in faces:
        # Extract the face region
        face_region = frame[face.top():face.bottom(), face.left():face.right()]

        # Check if face region is not empty before resizing
        if not face_region.size == 0:
            # Resize the face region to ensure consistent dimensions for encoding
            face_region_resized = cv2.resize(face_region, (128, 128))

            # Convert the face region to RGB format (required by face_recognition)
            face_region_rgb = cv2.cvtColor(face_region_resized, cv2.COLOR_BGR2RGB)

            # Extract the face encoding if faces are found
            face_encodings = face_recognition.face_encodings(face_region_rgb)
            if face_encodings:
                detected_face_encoding = face_encodings[0]

                # Assign face landmarks
                face_landmarks = dlib_facelandmark(gray, face)

                # Check if the detected face matches the driver's face
                if is_driver_face(detected_face_encoding, driver_face_encoding):
                    # Function to calculate eye aspect ratio (EAR)
                    def calculate_EAR(eye):
                        A = distance.euclidean(eye[1], eye[5])
                        B = distance.euclidean(eye[2], eye[4])
                        C = distance.euclidean(eye[0], eye[3])
                        eye_aspect_ratio = (A + B) / (2.0 * C)
                        return eye_aspect_ratio

                    # Face landmarks for eyes
                    leftEye = [(face_landmarks.part(n).x, face_landmarks.part(n).y) for n in range(36, 42)]
                    rightEye = [(face_landmarks.part(n).x, face_landmarks.part(n).y) for n in range(42, 48)]

                    left_ear = calculate_EAR(leftEye)
                    right_ear = calculate_EAR(rightEye)

                    EAR = (left_ear + right_ear) / 2
                    EAR = round(EAR, 2)

                    if EAR < 0.20:
                        cv2.putText(frame, "DROWSY", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 4)
                        cv2.putText(frame, "Take a break", (20, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                        print("Drowsy")
                        counter += 1
                        print(counter)

                        if counter >= 3:
                            print("Drowsy")
                            pygame.mixer.music.play()  # Play the custom sound when drowsiness is detected
                            break

                        time.sleep(1)

                    else:
                        counter = 0
                        print("Not Drowsy")

                    print(EAR)

                    # Draw a rectangle around the detected face
                    cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
                else:
                    # If the face does not match, display a message or take appropriate action
                    cv2.putText(frame, "Unauthorized Driver", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    pygame.mixer.music.play()  # Play the custom sound for unauthorized driver

    cv2.imshow("Driver's Face Recognition", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
