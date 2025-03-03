import os
import cv2
import numpy as np
from flask import Flask, request, render_template, url_for
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load your trained model
model = load_model('emotion_model.h5')

# Define emotion labels
emotion_labels = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

# Load OpenCV's Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def preprocess_face(face):
    # Ensure grayscale for consistent input to the model
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

    # Convert to grayscale for face detection
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Histogram Equalization for better contrast
    gray_image = cv2.equalizeHist(gray_image)

    # Detect faces with improved parameters
    faces = face_cascade.detectMultiScale(
        gray_image,
        scaleFactor=1.1,
        minNeighbors=8,  # Increased minNeighbors to reduce false positives
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    detected_emotions = []
    confidence_threshold = 0.5  # Increased confidence threshold

    for (x, y, w, h) in faces:
        if w < 30 or h < 30:  # Filter out small detections
            continue
        
        face_roi = gray_image[y:y+h, x:x+w]

        # Check if the detected face region is not too blank
        if np.sum(face_roi) == 0:
            continue
        
        processed_face = preprocess_face(face_roi)

        predictions = model.predict(processed_face)
        emotion_index = np.argmax(predictions[0])
        emotion_probability = np.max(predictions[0])
        
        # Log predictions for debugging
        print(f"Predictions: {predictions[0]}, Emotion Index: {emotion_index}, Probability: {emotion_probability}")

        # Only consider predictions above the confidence threshold
        if emotion_probability > confidence_threshold:
            emotion_text = f"{emotion_labels[emotion_index]} ({emotion_probability * 100:.1f}%)"
            detected_emotions.append(emotion_text)

            # Draw circle and label
            center = (x + w // 2, y + h // 2)  # Center of the face
            radius = int(max(w, h) / 2)  # Radius of the circle
            cv2.circle(image, center, radius, (0, 255, 0), 2)  # Draw the circle
            
            # Position for the emotion text
            y_pos = max(y - 10, 20)
            cv2.putText(image, emotion_text, (x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    os.makedirs('static/uploads', exist_ok=True)
    annotated_image_path = 'static/uploads/annotated_image.jpg'
    cv2.imwrite(annotated_image_path, image)

    return detected_emotions, annotated_image_path

@app.route('/')
def home():
    return render_template('imageanalysis.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file uploaded'
    
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    
    os.makedirs('static/uploads', exist_ok=True)
    file_path = os.path.join('static/uploads', 'input_image.jpg')
    file.save(file_path)

    detected_emotions, annotated_image_path = preprocess_image(file_path)
    
    if detected_emotions:
        return render_template(
            'imageanalysis.html',
            image_path=url_for('static', filename='uploads/annotated_image.jpg'),
            emotions=detected_emotions
        )
    else:
        return render_template(
            'imageanalysis.html',
            image_path=url_for('static', filename='uploads/annotated_image.jpg'),
            emotions=["No faces detected"]
        )

if __name__ == '__main__':
    app.run(debug=True)