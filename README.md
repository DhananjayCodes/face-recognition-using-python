# face-recognition-using-python
import cv2
import os
import pickle
import numpy as np
import time

class SimpleFaceAuth:
    def __init__(self, data_file="face_data.pickle"):
        self.data_file = data_file
        self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.faces = []
        self.labels = []
        self.names = []
        self.load_data()
        
    def load_data(self):
        if os.path.exists(self.data_file):
            with open(self.data_file, 'rb') as f:
                data = pickle.load(f)
                self.faces = data.get('faces', [])
                self.labels = data.get('labels', [])
                self.names = data.get('names', [])
                
            if self.faces and self.labels:
                self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
                faces_np = np.array(self.faces)
                labels_np = np.array(self.labels)
                self.face_recognizer.train(faces_np, labels_np)
                print(f"Loaded {len(self.names)} faces")
        else:
            print("No existing face data found")
    
    def save_data(self):
        data = {
            'faces': self.faces,
            'labels': self.labels,
            'names': self.names
        }
        with open(self.data_file, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved {len(self.names)} faces")
    
    def register_face(self):
        name = input("Enter your name: ")
        
        cap = cv2.VideoCapture(0)
        face_samples = []
        
        print("Capturing face. Look at the camera and move slightly for different angles.")
        print("Press 'q' to quit capturing")
        
        sample_count = 0
        
        while sample_count < 30:  # Collect 30 face samples
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Normalize face
                face_img = gray[y:y+h, x:x+w]
                face_img = cv2.resize(face_img, (100, 100))
                
                # Save face
                if len(faces) == 1:  # Only save if one face is detected
                    face_samples.append(face_img)
                    sample_count += 1
                    
            # Display progress
            cv2.putText(frame, f"Samples: {sample_count}/30", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Face Registration', frame)
            
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
                
            # Wait a bit between captures
            time.sleep(0.1)
        
        cap.release()
        cv2.destroyAllWindows()
        
        if sample_count > 0:
            # Create a unique ID for this person
            if not self.labels:
                new_label = 0
            else:
                new_label = max(self.labels) + 1
                
            # Add all samples with the same label
            for face in face_samples:
                self.faces.append(face)
                self.labels.append(new_label)
                
            self.names.append(name)
            
            # Train the recognizer with updated data
            faces_np = np.array(self.faces)
            labels_np = np.array(self.labels)
            self.face_recognizer.train(faces_np, labels_np)
            
            self.save_data()
            print(f"Successfully registered {name}")
            return True
        else:
            print("Failed to capture enough face samples")
            return False
    
    def authenticate(self, confidence_threshold=70):
        if not self.faces or not self.labels:
            print("No registered faces. Please register first.")
            return False
            
        print("Starting authentication...")
        cap = cv2.VideoCapture(0)
        
        start_time = time.time()
        timeout = 20  # 20 seconds timeout
        
        while time.time() - start_time < timeout:
            ret, frame = cap.read()
            if not ret:
                continue
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                face_img = gray[y:y+h, x:x+w]
                face_img = cv2.resize(face_img, (100, 100))
                
                # Try to recognize
                label, confidence = self.face_recognizer.predict(face_img)
                
                # Lower confidence is better in OpenCV's LBPH recognizer
                if confidence < confidence_threshold:
                    name = self.names[self.labels.index(label)]
                    
                    # Display recognition
                    cv2.putText(frame, f"{name} ({100-confidence:.1f}%)", 
                               (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    cap.release()
                    cv2.destroyAllWindows()
                    print(f"Authentication successful: {name}")
                    return True
                else:
                    cv2.putText(frame, "Unknown", (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Show remaining time
            remaining = timeout - int(time.time() - start_time)
            cv2.putText(frame, f"Time: {remaining}s", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.imshow('Authentication', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
        print("Authentication failed: No matching face found")
        return False

def main():
    auth_system = SimpleFaceAuth()
    
    while True:
        print("\n--- Face Authentication System ---")
        print("1. Register Face")
        print("2. Test Authentication")
        print("3. Exit")
        
        choice = input("Enter choice (1-3): ")
        
        if choice == '1':
            auth_system.register_face()
        elif choice == '2':
            auth_system.authenticate()
        elif choice == '3':
            break
        else:
            print("Invalid choice")

if __name__ == "__main__":
    main()
    
