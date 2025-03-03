import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, Response, redirect

app = Flask(__name__)

# Load the trained body language model
model = joblib.load('body_language.pkl')

# Initialize MediaPipe Holistic model
mp_holistic = mp.solutions.holistic

# Function to process video frames and predict body language
def generate_frames():
    cap = cv2.VideoCapture(0)

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Recolor feed
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)

            # Recolor image back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Extract and predict body language
            try:
                pose_landmarks = results.pose_landmarks.landmark
                face_landmarks = results.face_landmarks.landmark
                pose_row = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in pose_landmarks]).flatten()
                face_row = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in face_landmarks]).flatten()
                row = np.concatenate([pose_row, face_row])

                # Predict class
                X = pd.DataFrame([row])
                body_language_class = model.predict(X)[0]

                # Draw a small box in the upper-left corner for the predicted class
                cv2.rectangle(image, (10, 10), (300, 60), (245, 117, 16), -1)  # Box color and position
                cv2.putText(image, f'Body Language: {body_language_class}', (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)  # Emotion text
            except Exception as e:
                print(f"Error in prediction: {e}")  # Log error for debugging

            # Convert the image frame to byte format for streaming
            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

# Route to display the video stream
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route for the homepage
@app.route('/')
def home():
    return redirect('/realtime')  # Redirect to the real-time detection page

# Route for real-time detection
@app.route('/realtime')
def realtime_detection():
    return render_template('realtime.html')  # Ensure this template exists

if __name__ == '__main__':
    app.run(debug=True)