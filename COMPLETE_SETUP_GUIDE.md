# Complete Setup Guide - MUDRA ISL Detection System

## 🚨 CRITICAL FIXES APPLIED

This document explains the issues found and how they were fixed.

### Issues Identified and Fixed

1. **Model-Label Mismatch (FIXED)**
   - **Previous Issue**: The TensorFlow model had 6 output classes but labels.txt contained 20 labels
   - **Temporary Fix**: Labels were reduced to 6, but this reduced accuracy
   - **Current Fix**: Using the original keras_model.h5 (20 classes) with all 20 labels for better accuracy:
     - 0: Namaste
     - 1: Good Morning  
     - 2: Where?
     - 3: Sorry
     - 4: Thirsty
     - 5: Eat
     - 6: Thank You
     - 7: Yes
     - 8: No
     - 9: Please
     - 10: Help
     - 11: Good
     - 12: Bad
     - 13: Stop
     - 14: Go
     - 15: Come
     - 16: Sit
     - 17: Stand
     - 18: Hello
     - 19: Goodbye

2. **TensorFlow/Keras Compatibility Issue**
   - **Problem**: The model was saved in older SavedModel format incompatible with Keras 3
   - **Impact**: Model loading failures or incorrect predictions
   - **Fix**: Added `os.environ['TF_USE_LEGACY_KERAS'] = '1'` at the start of app.py and added `tf_keras` to requirements.txt

3. **Missing Environment Configuration**
   - **Problem**: No .env.example template for users to configure Supabase
   - **Fix**: Created .env.example with clear instructions

## 📋 System Requirements

- Python 3.8 or higher (tested with 3.12)
- Webcam (built-in or external)
- Internet connection for Supabase and Google Drive features
- 4GB RAM minimum
- Modern web browser (Chrome, Firefox, Edge recommended)

## 🔧 Complete Installation Steps

### 1. Clone the Repository
```bash
git clone <repository-url>
cd ann-pyproj
```

### 2. Create Virtual Environment (Recommended)
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

**Note**: Installation may take 5-10 minutes due to TensorFlow size (~500MB).

### 4. Configure Environment Variables

```bash
# Copy the example file
cp .env.example .env

# Edit .env with your preferred text editor
nano .env  # or vim, code, notepad, etc.
```

Add your Supabase credentials:
```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key-here
```

**Where to get Supabase credentials:**
1. Go to https://app.supabase.com
2. Select your project (or create a new one)
3. Go to Settings → API
4. Copy "Project URL" → paste as SUPABASE_URL
5. Copy "anon public" key → paste as SUPABASE_KEY

### 5. Set Up Supabase Database

1. Open Supabase SQL Editor: https://app.supabase.com/project/YOUR_PROJECT/sql
2. Copy and paste the contents of `supabase_setup.sql`
3. Click "Run" to create the `recent_detections` table

### 6. (Optional) Set Up Google Drive Upload

Follow instructions in `SETUP_GOOGLE_DRIVE.md` if you want to enable dataset upload functionality.

## 🚀 Running the Application

### Start the Server
```bash
python app.py
```

You should see output like:
```
WARNING:tensorflow:SavedModel saved prior to TF 2.5 detected...
WARNING:tensorflow:No training configuration found in save file...
 * Serving Flask app 'app'
 * Debug mode: on
 * Running on http://127.0.0.1:5000
```

**Important**: The TensorFlow warnings are normal and expected with this model.

### Access the Application
Open your browser and go to:
```
http://127.0.0.1:5000
```

## 📖 How to Use

### Detection & Output Tab
1. Click **"Start Detection"** button
2. Allow browser camera access when prompted
3. Show hand gestures to the camera
4. Detected signs appear in the "Text Translation" panel
5. Click **"Click to Speak Last Sentence"** to hear the detected sign

**Tips for better detection:**
- Ensure good lighting
- Position your hand clearly in front of the camera
- Hold gestures steady for 1-2 seconds
- Keep background simple/uncluttered
- Camera should be at chest/shoulder height

### Dataset Upload Tab
Upload training images/videos to Google Drive (requires Google Drive setup).

### Results & Metrics Tab
View detection history, statistics, and metrics:
- Total detections count
- Average confidence score
- Most frequently detected sign
- Recent detection history

## 🐛 Troubleshooting

### Camera Not Working

**Browser Permission Issues:**
```
Error: Camera access denied
```
**Fix**: 
- Chrome: Settings → Privacy and Security → Site Settings → Camera → Allow
- Firefox: Preferences → Privacy & Security → Permissions → Camera → Allow
- Ensure no other application is using the camera

**Hardware Issues:**
```bash
# Linux: Check camera availability
ls /dev/video*

# Should show: /dev/video0 or similar
```

### TensorFlow Warnings

```
WARNING:tensorflow:SavedModel saved prior to TF 2.5 detected...
```
**Status**: ✅ Normal - This warning is expected and does not affect functionality.

```
WARNING:tensorflow:No training configuration found in save file...
```
**Status**: ✅ Normal - Model is inference-only, this is expected.

### Model Loading Errors

```
ValueError: File format not supported: filepath=model.savedmodel
```
**Fix**: Ensure `TF_USE_LEGACY_KERAS=1` is set in app.py (should already be fixed).

### Incorrect Gesture Detection

**Symptoms**: Different word shown for different gestures, random predictions

**Root Cause**: Model-label mismatch (now fixed)

**Verification**:
```bash
python3 << 'EOF'
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
import tensorflow as tf

model = tf.keras.models.load_model('model.savedmodel')
with open('labels.txt', 'r') as f:
    labels = f.readlines()

print(f"Model output classes: {model.output_shape[-1]}")
print(f"Labels in file: {len(labels)}")
print("Match:", "✅ Yes" if model.output_shape[-1] == len(labels) else "❌ No")
EOF
```

Expected output:
```
Model output classes: 6
Labels in file: 6
Match: ✅ Yes
```

### Supabase Connection Errors

```
Error: supabase is not defined / Invalid credentials
```

**Fix**:
1. Verify .env file exists and has correct credentials
2. Check Supabase project is active (not paused)
3. Verify API key is the "anon public" key, not service role key
4. Ensure database table was created (run supabase_setup.sql)

### Google Drive Upload Fails

```
Error: credentials.json not found
```
**Fix**: Follow SETUP_GOOGLE_DRIVE.md to create OAuth credentials

```
Error: Invalid folder ID or no access to folder
```
**Fix**: 
- Leave folder ID blank to upload to root directory
- If using folder ID, ensure you have write access
- Extract folder ID from URL: `drive.google.com/drive/folders/[FOLDER_ID]`

## 🔍 Understanding the Codebase

### Project Structure
```
ann-pyproj/
├── app.py                    # Flask backend server
├── model.savedmodel/         # TensorFlow model (6 classes)
├── labels.txt                # Class labels (6 entries)
├── requirements.txt          # Python dependencies
├── .env.example             # Environment variables template
├── .env                     # Your config (create from .env.example)
├── templates/
│   └── index.html          # Frontend UI
├── static/
│   ├── script.js           # Frontend JavaScript
│   └── style.css           # Custom styles
└── temp_uploads/           # Temporary file storage
```

### How It Works

1. **Model Loading** (app.py, lines 30-33)
   - Loads TensorFlow SavedModel with 6 output classes
   - Uses legacy Keras for compatibility
   - Parses labels.txt to get class names

2. **Video Processing** (app.py, generate_frames function)
   - Captures webcam frames at ~30 fps
   - Resizes to 224x224 (model input size)
   - Normalizes pixel values: `(pixel / 127.5) - 1` → range [-1, 1]
   - Runs inference to get predictions

3. **Prediction Logic**
   - Model outputs softmax probabilities for 6 classes
   - Selects class with highest probability (argmax)
   - Maps index to label from labels.txt
   - Only saves to database if confidence > 70%

4. **Frontend** (templates/index.html, static/script.js)
   - Polls `/predict` endpoint every 500ms
   - Updates UI with detected sign and confidence
   - Provides text-to-speech output
   - Stores local detection history in localStorage

### Key Functions

**app.py:**
- `start_camera()`: Initializes webcam capture
- `stop_camera()`: Releases webcam resources  
- `generate_frames()`: Main detection loop
- `predict()`: Returns latest prediction to frontend
- `upload_dataset()`: Handles Google Drive uploads
- `get_recent_detections()`: Fetches detection history from Supabase

**script.js:**
- `setupDetection()`: Handles start/stop detection
- `saveDetection()`: Saves to localStorage
- `updateMetrics()`: Calculates statistics
- `displayHistory()`: Renders detection history

## 🎯 Model Information

- **Architecture**: MobileNetV2-based (likely from Teachable Machine)
- **Input**: 224x224x3 RGB images
- **Output**: 6 classes (softmax probabilities)
- **Classes**: Namaste, Good Morning, Where?, Sorry, Thirsty, Eat
- **Format**: TensorFlow SavedModel (legacy)

## 💡 Tips for Extending

### Adding More Gestures

To train a model with more gestures:

1. Use [Google Teachable Machine](https://teachablemachine.withgoogle.com/)
2. Create classes for your gestures (recommend 6-10 classes max)
3. Collect 50-100 images per gesture
4. Train and export as "TensorFlow" → "Keras"
5. Download and replace `model.savedmodel/`
6. Update `labels.txt` with your class names (format: `0 ClassName`)

### Improving Accuracy

1. **Better Training Data**:
   - Vary lighting conditions
   - Multiple hand positions/angles
   - Different backgrounds
   - Multiple people's hands

2. **Data Augmentation**:
   - Teachable Machine includes this automatically
   - Flips, rotations, brightness variations

3. **More Training Examples**:
   - Minimum 50 per class
   - Ideal: 100-200 per class

## 📚 Additional Resources

- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Supabase Documentation](https://supabase.com/docs)
- [OpenCV Python Tutorials](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
- [Google Teachable Machine](https://teachablemachine.withgoogle.com/)

## 🆘 Getting Help

If you encounter issues not covered here:

1. Check all warnings/errors in terminal output
2. Verify Python version: `python --version` (need 3.8+)
3. Verify dependencies installed: `pip list | grep -E "tensorflow|flask|opencv"`
4. Test model loading independently (see verification script above)
5. Check browser console for JavaScript errors (F12 → Console tab)

## 📝 Notes

- Detection history is stored in browser localStorage (persists across sessions)
- Supabase logging only saves detections with >70% confidence
- Google Drive upload is optional - detection works without it
- Model predictions run on CPU (GPU not required but would be faster)
- Application is designed for single-user local use

## ✅ Success Checklist

Before reporting issues, verify:
- [ ] Python 3.8+ installed
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] `.env` file created with valid Supabase credentials
- [ ] Supabase table created (ran supabase_setup.sql)
- [ ] Camera accessible and not used by other apps
- [ ] Browser has camera permissions enabled
- [ ] `labels.txt` has exactly 6 lines (not 20)
- [ ] App starts without errors (warnings are OK)
- [ ] Can access http://127.0.0.1:5000 in browser
