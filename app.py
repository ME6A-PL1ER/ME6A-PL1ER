import cv2
import numpy as np
import cupy as cp
import sqlite3
import os
from deepface import DeepFace

def initialize_database():
    conn = sqlite3.connect("faces.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS faces (id INTEGER PRIMARY KEY, name TEXT, encoding BLOB)''')
    conn.commit()
    conn.close()

def save_face_to_database(name, encoding):
    conn = sqlite3.connect("faces.db")
    c = conn.cursor()
    c.execute("INSERT INTO faces (name, encoding) VALUES (?, ?)", (name, encoding.tobytes()))
    conn.commit()
    conn.close()

def load_faces_from_database():
    conn = sqlite3.connect("faces.db")
    c = conn.cursor()
    c.execute("SELECT name, encoding FROM faces")
    data = c.fetchall()
    conn.close()
    known_names = []
    known_encodings = []
    for name, encoding in data:
        known_names.append(name)
        known_encodings.append(np.frombuffer(encoding, dtype=np.float64))
    return known_names, known_encodings

def list_cameras():
    cameras = []
    for index in range(10):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            cameras.append(index)
            cap.release()
    return cameras

def get_camera_specs(index):
    cap = cv2.VideoCapture(index)
    specs = {
        "Max FPS": int(cap.get(cv2.CAP_PROP_FPS)),
        "Resolution": (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))}
    cap.release()
    return specs

def main():
    initialize_database()
    print("Available cameras:")
    cameras = list_cameras()
    if not cameras:
        print("No cameras found.")
        return

    for idx in cameras:
        print(f"{idx}: Camera {idx}")

    selected_camera = int(input("Select camera by index: "))
    specs = get_camera_specs(selected_camera)
    print(f"Selected Camera Specs: {specs}")

    fps = specs["Max FPS"]
    resolution = specs["Resolution"]

    known_names, known_encodings = load_faces_from_database()

    video_capture = cv2.VideoCapture(selected_camera)
    video_capture.set(cv2.CAP_PROP_FPS, fps)
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

    font = cv2.FONT_HERSHEY_DUPLEX
    typing_name = ""
    new_face_detected = False
    face_encoding_to_save = None
    training_mode = False

    while True:
        ret, frame = video_capture.read()

        if not ret:
            print("Failed to grab frame")
            break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            # Fast face detection via OpenCV or other methods, then use CuPy for GPU-accelerated matrix operations
            faces = DeepFace.extract_faces(rgb_frame, detector_backend='mtcnn', enforce_detection=False)

            if faces:
                for face in faces:
                    print("Detected face:", face)  # Print the face object to inspect its structure
                    
                    name = "Unknown"

                    face_encoding = None  # Ensure face_encoding is initialized here

                    if len(known_names) > 0:
                        # Extract the face embedding
                        try:
                            face_encoding = face['embedding']  # Get the face embedding from DeepFace
                            face_encoding = cp.array(face_encoding)  # Move face embeddings to GPU with CuPy
                        except KeyError as e:
                            print("Error: 'embedding' key not found in the face object. Skipping this face.")
                            continue

                        for known_encoding, known_name in zip(known_encodings, known_names):
                            # Compute similarity (distance) in GPU space using CuPy
                            known_encoding = cp.array(known_encoding)
                            distance = cp.linalg.norm(face_encoding - known_encoding)
                            if distance < 0.6:  # Threshold for face match (you can adjust this)
                                name = known_name
                                break

                    if name == "Unknown":
                        if not new_face_detected:
                            new_face_detected = True
                            face_encoding_to_save = cp.array(face_encoding)  # Keep the face embedding on GPU
                            typing_name = ""

                    cv2.rectangle(frame, (face['facial_area'][0], face['facial_area'][1]), 
                                  (face['facial_area'][2], face['facial_area'][3]), (0, 255, 0), 2)
                    cv2.putText(frame, "New Face Detected!", (face['facial_area'][0], face['facial_area'][1] - 30), font, 0.5, (255, 255, 255), 1)
                    cv2.putText(frame, f"Name: {typing_name}|", (face['facial_area'][0], face['facial_area'][1] - 10), font, 0.5, (255, 255, 255), 1)

                    break

                cv2.putText(frame, name, (face['facial_area'][0], face['facial_area'][1] - 10), font, 0.5, (255, 255, 255), 1)

                print(f"Face detected: {name}, Location: {face['facial_area']}")

                if training_mode and name != "Unknown" and face_encoding_to_save is not None:
                    save_face_to_database(name, face_encoding_to_save)
                    known_names.append(name)
                    known_encodings.append(face_encoding_to_save)

        except ValueError as e:
            print(f"Error during face extraction: {e}")

        cv2.imshow("Video", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord("t"):
            training_mode = not training_mode
            if training_mode:
                print("Training mode enabled!")
            else:
                print("Training mode disabled!")

        if new_face_detected:
            if key == 8:  # Backspace
                typing_name = typing_name[:-1]
            elif key == 13:  # Enter
                save_face_to_database(typing_name, face_encoding_to_save)
                known_names.append(typing_name)
                known_encodings.append(face_encoding_to_save)
                new_face_detected = False
            elif key != 255:  # Other keys
                typing_name += chr(key)

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
