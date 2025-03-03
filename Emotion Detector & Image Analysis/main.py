import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, Response, request, url_for, redirect
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)

# Load the trained emotion detection model
emotion_model = load_model('emotion_model.h5')
emotion_labels = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the trained body language model
body_language_model = joblib.load('body_language.pkl')
mp_holistic = mp.solutions.holistic

# Home route
@app.route('/')
def home():
    return render_template('home.html')  # Ensure this points to the right HTML

# Real-time detection route
@app.route('/realtime')
def realtime_detection():
    return render_template('realtime.html')

# Image analysis route
@app.route('/image_analysis', methods=['GET', 'POST'])
def image_analysis():
    if request.method == 'POST':
        # Check if file was uploaded
        if 'file' not in request.files:
            return render_template('imageanalysis.html', error='No file uploaded')
        
        file = request.files['file']
        if file.filename == '':
            return render_template('imageanalysis.html', error='No selected file')
        
        # Create upload directory and save file
        os.makedirs('static/uploads', exist_ok=True)
        file_path = os.path.join('static/uploads', 'input_image.jpg')
        file.save(file_path)

        # Process image and get results
        detected_emotions, annotated_image_path = preprocess_image(file_path)
        
        # Return results
        return render_template(
            'imageanalysis.html',
            image_path=url_for('static', filename='uploads/annotated_image.jpg'),
            emotions=detected_emotions
        )
    
    # GET request - just show the upload form
    return render_template('imageanalysis.html')
# Video feed route
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Image preprocessing for analysis
def preprocess_face(face):
    if len(face.shape) == 3:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    
    face = face.astype('float32') / 255.0
    face = cv2.resize(face, (48, 48))
    face = np.expand_dims(face, axis=-1)
    face = np.expand_dims(face, axis=0)
    return face

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return [], None

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.equalizeHist(gray_image)

    faces = face_cascade.detectMultiScale(
        gray_image,
        scaleFactor=1.1,
        minNeighbors=8,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    detected_emotions = []
    confidence_threshold = 0.5  

    for (x, y, w, h) in faces:
        if w < 30 or h < 30:
            continue
        
        face_roi = gray_image[y:y+h, x:x+w]
        if np.sum(face_roi) == 0:
            continue
        
        processed_face = preprocess_face(face_roi)

        predictions = emotion_model.predict(processed_face)
        emotion_index = np.argmax(predictions[0])
        emotion_probability = np.max(predictions[0])
        
        if emotion_probability > confidence_threshold:
            emotion_text = f"{emotion_labels[emotion_index]} ({emotion_probability * 100:.1f}%)"
            detected_emotions.append(emotion_text)

            center = (x + w // 2, y + h // 2)
            radius = int(max(w, h) / 2)
            cv2.circle(image, center, radius, (0, 255, 0), 2)
            y_pos = max(y - 10, 20)
            cv2.putText(image, emotion_text, (x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    os.makedirs('static/uploads', exist_ok=True)
    annotated_image_path = 'static/uploads/annotated_image.jpg'
    cv2.imwrite(annotated_image_path, image)

    return detected_emotions, annotated_image_path

# Generate frames for video feed
def generate_frames():
    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                pose_landmarks = results.pose_landmarks.landmark
                face_landmarks = results.face_landmarks.landmark
                pose_row = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in pose_landmarks]).flatten()
                face_row = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in face_landmarks]).flatten()
                row = np.concatenate([pose_row, face_row])

                X = pd.DataFrame([row])
                body_language_class = body_language_model.predict(X)[0]

                cv2.rectangle(image, (10, 10), (300, 60), (245, 117, 16), -1)
                cv2.putText(image, f'Body Language: {body_language_class}', (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            except Exception as e:
                print(f"Error in prediction: {e}")

            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

if __name__ == '__main__':
    app.run(debug=True)