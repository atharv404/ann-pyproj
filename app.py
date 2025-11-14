import os
# CRITICAL: Set this BEFORE importing TensorFlow to use legacy Keras
# This is required for compatibility with older SavedModel format
os.environ['TF_USE_LEGACY_KERAS'] = '1'

from flask import Flask, render_template, Response, jsonify, request
from flask_cors import CORS
import tensorflow as tf
import cv2
import numpy as np
import base64
import mediapipe as mp
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import pickle
from supabase import create_client, Client
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

app = Flask(__name__)   
CORS(app)

# Initialize Supabase client (optional - app works without it)
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")

if supabase_url and supabase_key:
    try:
        supabase: Client = create_client(supabase_url, supabase_key)
        print("✅ Supabase connected successfully")
    except Exception as e:
        print(f"⚠️  Warning: Could not connect to Supabase: {e}")
        print("   Detection will work but history won't be saved to database")
        supabase = None
else:
    print("⚠️  Warning: Supabase credentials not found in .env file")
    print("   Detection will work but history won't be saved to database")
    supabase = None

np.set_printoptions(suppress=True)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

model = tf.keras.models.load_model("keras_model.h5", compile=False)

with open("labels.txt", "r") as f:
    class_names = [line.strip().split(' ', 1)[1] for line in f.readlines()]

camera = None
last_prediction = {"class": "Waiting...", "confidence": 0}
last_saved_class = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start_camera():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
    return jsonify({"status": "started"})

@app.route('/stop', methods=['POST'])
def stop_camera():
    global camera
    if camera:
        camera.release()
        camera = None
    return jsonify({"status": "stopped"})

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/predict')
def predict():
    return jsonify(last_prediction)

@app.route('/recent_detections', methods=['GET'])
def get_recent_detections():
    if not supabase:
        return jsonify({"status": "success", "data": []})
    try:
        result = supabase.table('recent_detections').select('*').order('timestamp', desc=True).limit(10).execute()
        return jsonify({"status": "success", "data": result.data})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/upload_dataset', methods=['POST'])
def upload_dataset():
    try:
        SCOPES = ['https://www.googleapis.com/auth/drive']
        creds = None
        if os.path.exists('token.pickle'):
            with open('token.pickle', 'rb') as token:
                creds = pickle.load(token)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not os.path.exists('credentials.json'):
                    return jsonify({"error": "credentials.json not found. Please add Google OAuth credentials."}), 400
                flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
                creds = flow.run_local_server(port=0)
            with open('token.pickle', 'wb') as token:
                pickle.dump(creds, token)
        
        service = build('drive', 'v3', credentials=creds)
        folder_id = request.form.get('folder_id', '').strip()
        
        if 'drive.google.com' in folder_id:
            folder_id = folder_id.split('/folders/')[-1].split('?')[0]
        
        if folder_id:
            try:
                service.files().get(fileId=folder_id, fields='id').execute()
            except:
                return jsonify({"error": "Invalid folder ID or no access to folder. Leave empty to upload to root."}), 400
        
        uploaded_files = []
        for file in request.files.getlist('files'):
            temp_path = os.path.join('temp_uploads', file.filename)
            os.makedirs('temp_uploads', exist_ok=True)
            file.save(temp_path)
            
            file_metadata = {'name': file.filename}
            if folder_id:
                file_metadata['parents'] = [folder_id]
            
            media = MediaFileUpload(temp_path, resumable=True)
            uploaded_file = service.files().create(body=file_metadata, media_body=media, fields='id,name,webViewLink').execute()
            uploaded_files.append(uploaded_file)
            
            try:
                os.remove(temp_path)
            except:
                pass
        
        return jsonify({"status": "success", "files": uploaded_files})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def generate_frames():
    global camera, last_prediction, last_saved_class
    while True:
        if camera is None:
            break
        ret, frame = camera.read()
        if not ret:
            break
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect hands
        results = hands.process(rgb_frame)
        
        # Default prediction when no hand is detected
        prediction_made = False
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Get bounding box coordinates
                h, w, c = frame.shape
                x_coords = [landmark.x * w for landmark in hand_landmarks.landmark]
                y_coords = [landmark.y * h for landmark in hand_landmarks.landmark]
                
                x_min, x_max = int(min(x_coords)), int(max(x_coords))
                y_min, y_max = int(min(y_coords)), int(max(y_coords))
                
                # Add padding around the hand
                padding = 20
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(w, x_max + padding)
                y_max = min(h, y_max + padding)
                
                # Draw bounding box (green rectangle)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                
                # Extract hand region
                hand_region = frame[y_min:y_max, x_min:x_max]
                
                if hand_region.size > 0:
                    # Preprocess hand region for model
                    try:
                        image = cv2.resize(hand_region, (224, 224), interpolation=cv2.INTER_AREA)
                        img_array = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
                        img_array = (img_array / 127.5) - 1
                        
                        # Make prediction
                        prediction = model.predict(img_array, verbose=0)
                        prediction_array = prediction.flatten()
                        
                        index = np.argmax(prediction_array)
                        class_name = class_names[index]
                        confidence = float(np.round(prediction_array[index] * 100, 2))
                        
                        # Only update prediction if confidence is above threshold
                        if confidence > 60:
                            last_prediction = {"class": class_name, "confidence": confidence}
                            prediction_made = True
                            
                            # Display prediction on frame
                            label = f"{class_name} ({confidence}%)"
                            cv2.putText(frame, label, (x_min, y_min - 10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            
                            # Save to database if high confidence and new detection
                            if supabase and class_name != last_saved_class and confidence > 70:
                                try:
                                    supabase.table('recent_detections').insert({
                                        "class_name": class_name,
                                        "confidence": confidence,
                                        "timestamp": datetime.utcnow().isoformat()
                                    }).execute()
                                    last_saved_class = class_name
                                except:
                                    pass
                    except Exception as e:
                        pass
        
        # If no hand detected or no valid prediction, show waiting message
        if not prediction_made and results.multi_hand_landmarks is None:
            last_prediction = {"class": "No hand detected", "confidence": 0}
        elif not prediction_made:
            last_prediction = {"class": "Low confidence", "confidence": 0}
        
        # Encode and stream frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
