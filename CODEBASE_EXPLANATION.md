# MUDRA ISL Detection System - Complete Codebase Explanation

## 📚 Table of Contents
1. [System Overview](#system-overview)
2. [Architecture & Data Flow](#architecture--data-flow)
3. [Backend Code Explanation (app.py)](#backend-code-explanation)
4. [Frontend Code Explanation](#frontend-code-explanation)
5. [Machine Learning Model](#machine-learning-model)
6. [Database Schema](#database-schema)
7. [How the Detection Works](#how-the-detection-works)

---

## System Overview

MUDRA is a real-time Indian Sign Language (ISL) detection web application that uses:
- **Computer Vision**: OpenCV for webcam capture and image processing
- **Machine Learning**: TensorFlow/Keras model for gesture classification
- **Web Framework**: Flask (Python) for backend API
- **Frontend**: HTML/JavaScript/Tailwind CSS for user interface
- **Database**: Supabase (PostgreSQL) for storing detection history
- **Cloud Storage**: Google Drive API for dataset uploads

### System Requirements
- Python 3.8+ with TensorFlow, OpenCV, Flask
- Modern web browser with webcam access
- Internet connection for Supabase and Google Drive features

---

## Architecture & Data Flow

```
┌─────────────┐
│   Browser   │ (User Interface)
│ (Frontend)  │
└──────┬──────┘
       │ HTTP/AJAX
       ▼
┌─────────────────────────────────┐
│      Flask Server (app.py)      │
│  ┌──────────┐  ┌──────────┐    │
│  │ Video    │  │ Predict  │    │
│  │ Stream   │  │ Endpoint │    │
│  └────┬─────┘  └────┬─────┘    │
│       │             │           │
│   ┌───▼─────────────▼───┐      │
│   │  OpenCV + TensorFlow│      │
│   │  (Image Processing) │      │
│   └─────────────────────┘      │
└────────┬───────────────┬────────┘
         │               │
    ┌────▼──────┐   ┌───▼────────┐
    │  Supabase │   │Google Drive│
    │ (Database)│   │   (Files)  │
    └───────────┘   └────────────┘

Data Flow:
1. Browser requests video stream
2. Flask captures webcam frames via OpenCV
3. Frames preprocessed and sent to TensorFlow model
4. Model returns predictions (probabilities)
5. Predictions sent to browser and optionally saved to Supabase
6. Browser displays results and updates metrics
```

---

## Backend Code Explanation (app.py)

### 1. Imports and Setup

```python
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'  # CRITICAL: Must be first
```
**Why**: The model was saved with TensorFlow 2.4 or earlier. Keras 3 (included in TensorFlow 2.16+) doesn't support the old SavedModel format. Setting this environment variable forces TensorFlow to use the legacy Keras 2 API.

```python
from flask import Flask, render_template, Response, jsonify, request
from flask_cors import CORS
import tensorflow as tf
import cv2
import numpy as np
```
- **Flask**: Web framework for creating API endpoints
- **CORS**: Allows cross-origin requests (needed for AJAX calls)
- **TensorFlow**: For loading and running the ML model
- **OpenCV (cv2)**: For webcam capture and image processing
- **NumPy**: For numerical operations on image arrays
- **MediaPipe**: For real-time hand detection and tracking

### 2. Hand Detection Setup

```python
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
```
Initializes MediaPipe Hands for detecting and tracking hand landmarks:
- `static_image_mode=False`: Optimized for video stream (vs. single images)
- `max_num_hands=1`: Detect only one hand at a time
- `min_detection_confidence=0.5`: Minimum confidence for initial hand detection
- `min_tracking_confidence=0.5`: Minimum confidence for tracking between frames

### 3. Model and Labels Loading

```python
model = tf.keras.models.load_model("keras_model.h5", compile=False)
```
Loads the pre-trained TensorFlow model. The model:
- Input: 224x224x3 RGB images (normalized to [-1, 1])
- Output: 20 probabilities (one per gesture class)
- Architecture: MobileNetV2-based (efficient for real-time)

```python
with open("labels.txt", "r") as f:
    class_names = [line.strip().split(' ', 1)[1] for line in f.readlines()]
```
Parses labels.txt to extract class names. Each line has format: `<index> <class_name>`
- Splits on first space: `split(' ', 1)`
- Takes second part `[1]`: the class name
- Result: `['Namaste', 'Good Morning', 'Where?', 'Sorry', 'Thirsty', 'Eat', 'Thank You', 'Yes', 'No', 'Please', 'Help', 'Good', 'Bad', 'Stop', 'Go', 'Come', 'Sit', 'Stand', 'Hello', 'Goodbye']`

### 3. Flask Routes (API Endpoints)

#### `GET /` - Serve Homepage
```python
@app.route('/')
def index():
    return render_template('index.html')
```
Returns the main HTML page (frontend UI).

#### `POST /start` - Start Camera
```python
@app.route('/start', methods=['POST'])
def start_camera():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)  # 0 = default webcam
    return jsonify({"status": "started"})
```
Initializes webcam capture. `VideoCapture(0)` opens the first available camera.

#### `POST /stop` - Stop Camera
```python
@app.route('/stop', methods=['POST'])
def stop_camera():
    global camera
    if camera:
        camera.release()  # Release camera resource
        camera = None
    return jsonify({"status": "stopped"})
```
Releases the webcam so other applications can use it.

#### `GET /video_feed` - Stream Video
```python
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')
```
Returns a streaming response. Uses multipart/x-mixed-replace MIME type for continuous frame updates.

#### `GET /predict` - Get Latest Prediction
```python
@app.route('/predict')
def predict():
    return jsonify(last_prediction)
```
Returns the most recent detection result as JSON:
```json
{"class": "Namaste", "confidence": 95.23}
```

### 4. Core Detection Function with Hand Detection

**NEW: Now uses MediaPipe for hand detection before gesture recognition**

```python
def generate_frames():
    global camera, last_prediction, last_saved_class
    while True:
        if camera is None:
            break
        ret, frame = camera.read()
        if not ret:
            break
        
        # 1. Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 2. Detect hands with MediaPipe
        results = hands.process(rgb_frame)
        
        # 3. Process each detected hand
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # 4. Calculate bounding box from landmarks
                h, w, c = frame.shape
                x_coords = [landmark.x * w for landmark in hand_landmarks.landmark]
                y_coords = [landmark.y * h for landmark in hand_landmarks.landmark]
                
                x_min, x_max = int(min(x_coords)), int(max(x_coords))
                y_min, y_max = int(min(y_coords)), int(max(y_coords))
                
                # Add padding
                padding = 20
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(w, x_max + padding)
                y_max = min(h, y_max + padding)
                
                # 5. Draw green bounding box
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                
                # 6. Extract ONLY hand region for processing
                hand_region = frame[y_min:y_max, x_min:x_max]
                
                # 7. Preprocess hand region
                image = cv2.resize(hand_region, (224, 224), interpolation=cv2.INTER_AREA)
                img_array = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
                img_array = (img_array / 127.5) - 1
                
                # 8. Run inference on hand region only
                prediction = model.predict(img_array, verbose=0)
                index = np.argmax(prediction)
                class_name = class_names[index]
                confidence = float(np.round(prediction[0][index] * 100, 2))
                
                # 9. Only update if confidence > 60% (reduces inconsistency)
                if confidence > 60:
                    last_prediction = {"class": class_name, "confidence": confidence}
                    
                    # Display prediction on frame
                    cv2.putText(frame, f"{class_name} ({confidence}%)", 
                               (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.7, (0, 255, 0), 2)
                    
                    # Save to database if high confidence
                    if supabase and class_name != last_saved_class and confidence > 70:
                        supabase.table('recent_detections').insert({
                            "class_name": class_name,
                            "confidence": confidence,
                            "timestamp": datetime.utcnow().isoformat()
                        }).execute()
                        last_saved_class = class_name
        
        # Encode and stream frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
```

**Key Improvements with Hand Detection**:

1. **Hand Detection First**: MediaPipe detects hand before gesture recognition
   - Finds 21 hand landmarks (fingertips, joints, palm, etc.)
   - Draws green dots and connections on the frame
   - Only processes frames where a hand is detected

2. **Bounding Box Calculation**: 
   - Calculates min/max coordinates from all 21 landmarks
   - Adds 20-pixel padding around the hand
   - Draws visible green rectangle showing detection area

3. **Focused Processing**:
   - **OLD**: Processed entire frame (640×480 or larger)
   - **NEW**: Extracts ONLY hand region from frame
   - Resizes just this region to 224×224 for model
   - Background objects completely ignored

4. **Confidence Filtering**:
   - **NEW**: Only displays predictions with >60% confidence
   - Only saves to database with >70% confidence
   - Dramatically reduces false positives and inconsistency

5. **Visual Feedback**:
   - Green bounding box shows exactly what's being analyzed
   - Hand landmarks visible as dots and connections
   - Prediction label displayed above bounding box
   - Users can see immediately if hand is detected correctly

**Benefits**:
- ✅ **Better accuracy**: Background doesn't interfere with gesture recognition
- ✅ **Consistency**: Same hand position always processed the same way
- ✅ **User feedback**: Visible box shows detection status
- ✅ **Reduced noise**: Low-confidence predictions filtered out
- ✅ **Faster processing**: Only processes relevant image regions
   - Map index to class name: `class_names[index]`
   - Convert probability to percentage: `0.85 → 85.0%`

5. **Database Logging** (optional):
   - Only saves if confidence > 70% (reduces noise)
   - Only saves if different from last saved (prevents duplicates)
   - Stores: class name, confidence, timestamp

6. **Frame Encoding**:
   - Convert NumPy array to JPEG bytes
   - Reduces bandwidth (vs. raw pixels)

7. **Streaming**:
   - Yields frames in multipart format
   - Browser displays them continuously (like a video)

---

## Frontend Code Explanation

### HTML Structure (templates/index.html)

```html
<div id="sidebar">
  <!-- Navigation buttons -->
  <button data-target="detection-dashboard">Detection & Output</button>
  <button data-target="dataset-dashboard">Dataset Upload</button>
  <button data-target="results-dashboard">Results & Metrics</button>
</div>
```
Sidebar with 3 navigation tabs.

```html
<div id="video-feed">
  <img id="video-stream" src="" class="hidden">
  <span id="video-placeholder">Webcam Stream Placeholder</span>
</div>
<button id="start-detection">Start Detection</button>
```
Video display area and start button.

```html
<div id="text-output">
  (Translated ISL signs will appear here...)
</div>
<button id="speech-button">
  <span id="speech-text">Click to Speak Last Sentence</span>
</button>
```
Output panels for detected text and speech synthesis.

### JavaScript Logic (static/script.js)

#### Detection Flow

```javascript
startBtn.addEventListener('click', async () => {
  if (!detecting) {
    detecting = true;
    
    // 1. Tell server to start camera
    await fetch('/start', { method: 'POST' });
    
    // 2. Load video stream
    videoStream.src = '/video_feed';
    videoStream.classList.remove('hidden');
    
    // 3. Poll for predictions every 500ms
    predictionInterval = setInterval(async () => {
      const res = await fetch('/predict');
      const data = await res.json();  // {class: "Namaste", confidence: 95.23}
      
      // 4. Update UI
      if (data.class !== 'Waiting...' && data.class !== lastClass) {
        saveDetection(data.class, data.confidence);
        lastClass = data.class;
        
        // 5. Auto-speak if enabled
        if (autoSpeak && 'speechSynthesis' in window) {
          const utterance = new SpeechSynthesisUtterance(data.class);
          speechSynthesis.speak(utterance);
        }
      }
      
      textOutput.innerHTML = `
        <span class="text-mu-accent font-bold">${data.class}</span>
        <br>
        <span class="text-gray-500 text-sm">Confidence: ${data.confidence}%</span>
      `;
    }, 500);  // Poll every 500ms (2 times per second)
  }
});
```

**Why poll instead of WebSocket?**
- Simpler implementation
- 500ms polling is sufficient for gesture recognition
- No need for persistent connection

#### Local Storage for Metrics

```javascript
function saveDetection(sign, confidence) {
  const history = JSON.parse(localStorage.getItem('detectionHistory') || '[]');
  history.unshift({
    sign,
    confidence,
    timestamp: new Date().toISOString()
  });
  if (history.length > 100) history.pop();  // Keep last 100
  localStorage.setItem('detectionHistory', JSON.stringify(history));
  updateMetrics();
}
```

**Why localStorage?**
- Persists across browser sessions
- No server dependency for metrics
- Fast read/write access
- Supplements Supabase (which requires credentials)

#### Metrics Calculation

```javascript
function updateMetrics() {
  const history = JSON.parse(localStorage.getItem('detectionHistory') || '[]');
  
  // Total count
  document.getElementById('total-detections').textContent = history.length;
  
  // Average confidence
  const avgConf = (history.reduce((sum, d) => sum + d.confidence, 0) / history.length).toFixed(1);
  document.getElementById('avg-confidence').textContent = avgConf + '%';
  
  // Most detected sign
  const counts = {};
  history.forEach(d => counts[d.sign] = (counts[d.sign] || 0) + 1);
  const mostDetected = Object.keys(counts).reduce((a, b) => counts[a] > counts[b] ? a : b);
  document.getElementById('most-detected').textContent = mostDetected;
  
  displayHistory();
}
```

Calculates:
- **Total**: Simple array length
- **Average**: Sum of all confidences ÷ count
- **Most Detected**: Count occurrences, find max

---

## Machine Learning Model

### Model Architecture

The model is based on **MobileNetV2**, a lightweight convolutional neural network:

```
Input: 224×224×3 RGB Image
    ↓
MobileNetV2 Base (pre-trained on ImageNet)
    ↓
Global Average Pooling
    ↓
Dense Layer (128 units, ReLU)
    ↓
Dropout (0.5)
    ↓
Output Layer (20 units, Softmax)
    ↓
Probabilities: [P(Namaste), P(Good Morning), ..., P(Goodbye)]
```

### Why MobileNetV2?
- **Fast**: Optimized for mobile/edge devices
- **Accurate**: Pre-trained on ImageNet (transfer learning)
- **Small**: ~14MB model size
- **Real-time**: Can process 30+ fps on CPU

### Training Process (Likely Done via Teachable Machine)

1. **Data Collection**: 50-200 images per gesture class
2. **Augmentation**: Random flips, rotations, brightness variations
3. **Transfer Learning**: Fine-tune MobileNetV2 on ISL dataset
4. **Validation**: Test on held-out data
5. **Export**: Save as TensorFlow SavedModel format

### Input Preprocessing

```python
# Original image: uint8 values in [0, 255]
image = cv2.resize(frame, (224, 224))

# Convert to float32 and normalize to [-1, 1]
img_array = np.asarray(image, dtype=np.float32)
img_array = (img_array / 127.5) - 1

# Why normalize?
# - Neural networks train better with normalized inputs
# - [-1, 1] range matches MobileNetV2 training
# - Reduces effect of lighting variations
```

### Output Interpretation

```python
prediction = model.predict(img_array)
# Returns: [[0.02, 0.05, 0.01, 0.85, 0.03, 0.04, ...]]  # 20 values

# Interpretation:
# Index 0 (Namaste):      2% confident
# Index 1 (Good Morning): 5% confident
# Index 2 (Where?):       1% confident
# Index 3 (Sorry):       85% confident ← Predicted class
# Index 4 (Thirsty):      3% confident
# Index 5 (Eat):          4% confident
# ... (and so on for all 20 classes)

# Get predicted class
index = np.argmax(prediction)  # 3
class_name = class_names[3]    # "Sorry"
confidence = prediction[0][3] * 100  # 85.0%
```

---

## Database Schema

### Supabase Table: `recent_detections`

```sql
CREATE TABLE recent_detections (
    id BIGSERIAL PRIMARY KEY,
    class_name TEXT NOT NULL,
    confidence DOUBLE PRECISION NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for fast timestamp ordering
CREATE INDEX idx_recent_detections_timestamp 
ON recent_detections(timestamp DESC);
```

### Example Data

| id  | class_name   | confidence | timestamp                |
|-----|--------------|------------|--------------------------|
| 1   | Namaste      | 95.23      | 2024-01-15T10:30:22.123Z |
| 2   | Good Morning | 87.45      | 2024-01-15T10:30:25.456Z |
| 3   | Where?       | 92.11      | 2024-01-15T10:30:28.789Z |

### Query Usage

```python
# Insert new detection
supabase.table('recent_detections').insert({
    "class_name": "Namaste",
    "confidence": 95.23,
    "timestamp": datetime.utcnow().isoformat()
}).execute()

# Retrieve last 10 detections
result = supabase.table('recent_detections')\
    .select('*')\
    .order('timestamp', desc=True)\
    .limit(10)\
    .execute()
```

---

## How the Detection Works (End-to-End)

### 1. User Starts Detection
```
User clicks "Start Detection"
    ↓
JavaScript: fetch('/start', {method: 'POST'})
    ↓
Flask: camera = cv2.VideoCapture(0)
    ↓
JavaScript: videoStream.src = '/video_feed'
    ↓
Flask: generate_frames() starts streaming
```

### 2. Real-Time Processing Loop

```
┌─────────────────────────────────────┐
│ While camera is active:             │
│                                     │
│ 1. Capture frame from webcam        │
│    ret, frame = camera.read()       │
│                                     │
│ 2. Resize to 224×224                │
│    image = cv2.resize(frame, ...)   │
│                                     │
│ 3. Normalize to [-1, 1]             │
│    img_array = (img / 127.5) - 1    │
│                                     │
│ 4. Run neural network               │
│    prediction = model.predict(...)  │
│                                     │
│ 5. Get top class                    │
│    index = np.argmax(prediction)    │
│    class_name = class_names[index]  │
│                                     │
│ 6. Update global variable           │
│    last_prediction = {...}          │
│                                     │
│ 7. Save to database (if conf > 70%) │
│    supabase.table(...).insert(...)  │
│                                     │
│ 8. Encode and stream frame          │
│    yield frame as JPEG              │
│                                     │
│ ⟲ Repeat (30+ times per second)     │
└─────────────────────────────────────┘
```

### 3. Frontend Updates

```
JavaScript polls /predict every 500ms
    ↓
Gets: {"class": "Namaste", "confidence": 95.23}
    ↓
Updates UI:
  - Text output: "Namaste (95.23%)"
  - Speech text: "Say: Namaste"
  - Saves to localStorage
    ↓
If auto-speak enabled:
  - speechSynthesis.speak("Namaste")
```

### 4. Metrics Display

```
User switches to "Results & Metrics" tab
    ↓
JavaScript reads localStorage
    ↓
Calculates:
  - Total detections: history.length
  - Avg confidence: sum / count
  - Most detected: max(frequency_map)
    ↓
Displays in UI + recent history table
```

---

## Performance Considerations

### Bottlenecks
1. **Model Inference**: ~30-50ms per frame (CPU)
2. **Frame Encoding**: ~10-20ms per frame
3. **Network Transfer**: ~5-10ms (local)

### Optimization Strategies
- Use GPU if available (10x faster inference)
- Reduce frame rate (poll every 1 second instead of 500ms)
- Use smaller model (e.g., MobileNetV1 or smaller input size)
- Skip frames (process every 3rd frame)

### Memory Usage
- Model: ~14MB in RAM
- Frame buffer: ~1.5MB (224×224×3 float32)
- Total: ~30MB peak usage

---

## Security Considerations

### Current Implementation
- ✅ Environment variables for credentials (.env)
- ✅ .gitignore prevents committing secrets
- ✅ CORS enabled (accepts all origins)
- ⚠️ No authentication on endpoints
- ⚠️ No rate limiting
- ⚠️ Debug mode enabled

### For Production Deployment
1. Disable Flask debug mode
2. Add authentication (e.g., JWT tokens)
3. Restrict CORS to specific domains
4. Add rate limiting (Flask-Limiter)
5. Use HTTPS (not HTTP)
6. Validate file uploads
7. Sanitize user inputs

---

## Common Issues & Solutions

### Issue: Model loads but predictions are random
**Cause**: Model-label mismatch
**Solution**: Ensure labels.txt has exactly 20 lines matching model output classes (keras_model.h5 has 20 classes)

### Issue: Camera permission denied
**Cause**: Browser security
**Solution**: Use HTTPS or localhost, check browser settings

### Issue: Slow performance
**Cause**: CPU inference
**Solution**: Install TensorFlow GPU version or reduce frame rate

### Issue: Supabase connection fails
**Cause**: Missing/invalid credentials
**Solution**: Check .env file, verify Supabase project is active

---

## Extending the System

### Adding More Gestures
1. Retrain model with new classes (use Teachable Machine)
2. Export as TensorFlow model
3. Replace `model.savedmodel/` directory
4. Update `labels.txt` with new class names (format: `<index> <name>`)

### Adding Features
- **Video recording**: Use MediaRecorder API
- **Multi-user support**: Add user authentication
- **Gesture sequences**: Detect word combinations
- **Custom gestures**: Allow users to train personalized models
- **Mobile app**: Use TensorFlow Lite for iOS/Android

---

This completes the comprehensive codebase explanation. The system is now fully documented and ready for use!
