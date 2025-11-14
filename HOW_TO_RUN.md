# 🚀 Quick Start Guide - Running MUDRA ISL Detection Locally

This guide provides **precise, step-by-step instructions** to run this project on your local machine.

---

## ⚡ Prerequisites Check

Before starting, verify you have:

```bash
# Check Python version (need 3.8 or higher)
python --version
# or
python3 --version

# Expected output: Python 3.8.x or higher
```

If Python is not installed or version is too old:
- **Windows**: Download from https://www.python.org/downloads/
- **macOS**: `brew install python3` or download from python.org
- **Linux**: `sudo apt install python3 python3-pip` (Ubuntu/Debian)

---

## 📥 Step 1: Get the Code

### Option A: Clone with Git
```bash
git clone https://github.com/atharv404/ann-pyproj.git
cd ann-pyproj
```

### Option B: Download ZIP
1. Go to https://github.com/atharv404/ann-pyproj
2. Click "Code" → "Download ZIP"
3. Extract the ZIP file
4. Open terminal/command prompt in the extracted folder

---

## 🔧 Step 2: Set Up Virtual Environment (Recommended)

### Why Use Virtual Environment?
- Keeps project dependencies isolated
- Prevents conflicts with other Python projects
- Easy to reset if something goes wrong

### Windows
```cmd
python -m venv venv
venv\Scripts\activate
```

### macOS/Linux
```bash
python3 -m venv venv
source venv/bin/activate
```

**You should see `(venv)` appear in your terminal prompt.**

To deactivate later (when done):
```bash
deactivate
```

---

## 📦 Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**This will take 5-10 minutes** as it downloads ~500MB of packages (TensorFlow is large).

Expected output:
```
Collecting flask
Collecting tensorflow
...
Successfully installed flask-3.1.2 tensorflow-2.20.0 opencv-python-4.12.0 ...
```

### Troubleshooting Installation

**Error: "pip: command not found"**
```bash
# Try pip3 instead
pip3 install -r requirements.txt
```

**Error: "Permission denied"**
```bash
# Add --user flag
pip install --user -r requirements.txt
```

**Error: "Failed building wheel for opencv-python"**
```bash
# Install system dependencies first (Linux only)
sudo apt-get install python3-dev libhdf5-dev
pip install -r requirements.txt
```

---

## ⚙️ Step 4: Configure Environment (Optional but Recommended)

The app works **without configuration** (detection only), but Supabase is needed to save history.

### Create .env file

```bash
# Copy the example file
cp .env.example .env

# Edit with your preferred editor
nano .env
# or
vim .env
# or open in your text editor
```

### Get Supabase Credentials (Free Tier)

1. **Sign up**: Go to https://supabase.com (free account)
2. **Create project**: Click "New Project"
   - Name: "mudra-isl" (or anything)
   - Database password: Choose a strong password (save it!)
   - Region: Choose closest to you
   - Click "Create new project" (takes ~2 minutes)

3. **Get credentials**:
   - Go to Settings (⚙️ icon) → API
   - Copy **"Project URL"** → Paste as `SUPABASE_URL`
   - Copy **"anon public"** key → Paste as `SUPABASE_KEY`

4. **Set up database table**:
   - Go to SQL Editor (🗂️ icon)
   - Open `supabase_setup.sql` from project folder
   - Copy entire contents
   - Paste in SQL Editor
   - Click "Run" (bottom right)
   - Should see: "Success. No rows returned"

Your `.env` should look like:
```env
SUPABASE_URL=https://abcdefgh12345.supabase.co
SUPABASE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

**Skip this step if you just want to test detection without saving history.**

---

## 🎬 Step 5: Run the Application

```bash
python app.py
```

### Expected Output

You should see:
```
⚠️  Warning: Supabase credentials not found in .env file
   Detection will work but history won't be saved to database
WARNING:tensorflow:SavedModel saved prior to TF 2.5 detected...
WARNING:tensorflow:No training configuration found in save file...
 * Serving Flask app 'app'
 * Debug mode: on
WARNING: This is a development server. Do not use it in a production deployment.
 * Running on http://127.0.0.1:5000
Press CTRL+C to quit
```

**Note**: The warnings are normal and expected. The app is working correctly!

### If You See Errors

**Error: "ModuleNotFoundError: No module named 'tensorflow'"**
```bash
# Dependencies not installed
pip install -r requirements.txt
```

**Error: "Address already in use"**
```bash
# Port 5000 is busy, use different port
python app.py --port 5001
# Then access http://127.0.0.1:5001
```

**Error: "No module named 'cv2'"**
```bash
# OpenCV not installed
pip install opencv-python
```

---

## 🌐 Step 6: Access the Application

1. **Open your web browser** (Chrome, Firefox, or Edge recommended)

2. **Go to**: http://127.0.0.1:5000

3. **You should see**: The MUDRA dashboard with three tabs

---

## 📸 Step 7: Test Detection

### First Time Setup (Browser Permissions)

1. Click **"Start Detection"** button
2. Browser will ask: **"Allow camera access?"**
   - Click **"Allow"** or **"Yes"**
3. You should see your webcam feed appear

### Using the App

1. **Detection Tab** (should be active by default):
   - Click "Start Detection"
   - Show hand gestures to camera
   - Detected sign appears in "Text Translation" panel
   - Confidence percentage shown below

2. **Available Gestures** (6 total):
   - Namaste
   - Good Morning
   - Where?
   - Sorry
   - Thirsty
   - Eat

3. **Text-to-Speech**:
   - Click "Click to Speak Last Sentence"
   - Computer will say the detected gesture aloud
   - Click again to toggle auto-speak mode

4. **Results Tab**:
   - View total detections
   - See average confidence
   - Check most frequently detected sign
   - Review detection history

---

## 🔍 Verification Tests

### Test 1: Model Loaded Correctly
```bash
python3 << 'EOF'
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
import tensorflow as tf

model = tf.keras.models.load_model('model.savedmodel')
with open('labels.txt', 'r') as f:
    labels = f.readlines()

print(f"Model classes: {model.output_shape[-1]}")
print(f"Label count: {len(labels)}")
print("Status:", "✅ PASS" if model.output_shape[-1] == len(labels) else "❌ FAIL")
EOF
```

Expected output:
```
Model classes: 6
Label count: 6
Status: ✅ PASS
```

### Test 2: Server Accessible
```bash
# In a new terminal (keep app running in first terminal)
curl http://127.0.0.1:5000
# Should return HTML content (the homepage)
```

### Test 3: Camera Detection
1. Start detection in browser
2. Open browser DevTools (F12)
3. Go to Console tab
4. Should see no errors
5. Look at Network tab → should see `/predict` requests every 500ms

---

## 🛑 Stopping the Application

### In Terminal
Press **Ctrl+C** to stop the Flask server

### In Browser
Click **"Stop Detection"** button to release camera

---

## 📂 Project File Structure

```
ann-pyproj/
├── app.py                      # Flask server (main application)
├── model.savedmodel/           # TensorFlow model files
│   ├── saved_model.pb         # Model architecture
│   └── variables/             # Model weights
├── labels.txt                  # 6 gesture class names
├── requirements.txt            # Python dependencies
├── .env.example               # Environment variables template
├── .env                       # Your configuration (create this)
├── templates/
│   └── index.html            # Frontend HTML
├── static/
│   ├── script.js             # Frontend JavaScript
│   └── style.css             # Custom styles
├── supabase_setup.sql        # Database schema
├── COMPLETE_SETUP_GUIDE.md   # Detailed troubleshooting guide
├── CODEBASE_EXPLANATION.md   # Code architecture documentation
└── HOW_TO_RUN.md             # This file
```

---

## 💡 Tips for Best Results

### Camera Position
- **Height**: Chest/shoulder level
- **Distance**: 2-3 feet from camera
- **Angle**: Camera facing you directly (not tilted)

### Lighting
- **Good**: Natural daylight or bright room lighting
- **Avoid**: Backlighting (window behind you)
- **Avoid**: Very dark rooms

### Hand Gestures
- **Hold steady**: Keep gesture still for 1-2 seconds
- **Clear view**: Ensure entire hand is visible
- **Background**: Simple/plain background works best
- **Speed**: Make gestures slowly and deliberately

### Performance
- **Good WiFi**: If using Supabase features
- **Close other tabs**: Browser performance matters
- **Modern browser**: Chrome/Edge/Firefox (latest versions)

---

## 🐛 Common Issues & Quick Fixes

### Issue: Camera not showing

**Solution 1**: Check browser permissions
- Chrome: Settings → Privacy → Site Settings → Camera → Allow
- Firefox: about:preferences#privacy → Permissions → Camera → Allow

**Solution 2**: Check if camera is in use
- Close Zoom, Skype, Teams, or other video apps
- Only one app can use camera at a time

**Solution 3**: Try different browser
- Chrome, Firefox, and Edge all work
- Safari may have issues

### Issue: Detection shows wrong gestures

**Cause**: Fixed in latest version!

**Verify fix**:
```bash
# Check labels.txt has exactly 6 lines
wc -l labels.txt
# Should output: 6 labels.txt

# Check app.py has legacy Keras setting
head -5 app.py | grep LEGACY_KERAS
# Should show: os.environ['TF_USE_LEGACY_KERAS'] = '1'
```

### Issue: Slow performance / Lag

**Solution 1**: Reduce polling frequency
- Edit `static/script.js`
- Find: `}, 500);`
- Change to: `}, 1000);` (poll every 1 second instead of 0.5)

**Solution 2**: Close other programs
- Browser uses significant CPU for ML
- Close unused tabs and applications

**Solution 3**: Use simpler background
- Plain wall works best
- Less for model to process

### Issue: "Module not found" errors

**Solution**:
```bash
# Reinstall all dependencies
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

---

## 🔄 Updating the Project

If you pull new changes from GitHub:

```bash
# Get latest code
git pull

# Reinstall dependencies (in case they changed)
pip install -r requirements.txt

# Restart the app
python app.py
```

---

## 🎯 Next Steps

Once you have the basic app running:

1. **Try all gestures**: Test each of the 6 supported signs
2. **Check metrics**: View your detection history in Results tab
3. **Read docs**: Check out `CODEBASE_EXPLANATION.md` to understand how it works
4. **Extend it**: See `COMPLETE_SETUP_GUIDE.md` for ideas on adding features

---

## 📞 Getting Help

If you're stuck:

1. **Check terminal output**: Look for error messages
2. **Check browser console**: Press F12 → Console tab
3. **Review guides**:
   - `COMPLETE_SETUP_GUIDE.md` - Detailed troubleshooting
   - `CODEBASE_EXPLANATION.md` - How the code works
4. **Verify system**:
   - Python version: `python --version`
   - Packages installed: `pip list`
   - App running: `curl http://127.0.0.1:5000`

---

## ✅ Success Checklist

- [ ] Python 3.8+ installed
- [ ] Virtual environment created and activated
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] App starts without errors (`python app.py`)
- [ ] Can access http://127.0.0.1:5000 in browser
- [ ] Camera feed shows when clicking "Start Detection"
- [ ] Browser granted camera permissions
- [ ] Detections appear in text output panel
- [ ] (Optional) Supabase configured and history saving

**If all boxes checked: ✅ You're all set!**

---

## 🎉 Congratulations!

You've successfully set up and run the MUDRA ISL Detection System!

**What you can do now:**
- Detect 6 different Indian Sign Language gestures in real-time
- View your detection history and statistics
- Use text-to-speech to hear detected gestures
- Learn about how ML-powered gesture recognition works

**Next learning steps:**
- Read the codebase documentation to understand the internals
- Try training your own custom gestures with Google Teachable Machine
- Extend the app with new features (see COMPLETE_SETUP_GUIDE.md)

Happy gesture detecting! 🙌
