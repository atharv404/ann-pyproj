# MUDRA - Indian Sign Language Detection System

A real-time Indian Sign Language (ISL) detection system using TensorFlow and Flask with automatic detection logging and Google Drive integration.

## 🚨 IMPORTANT: Recent Fixes

**Critical issues have been identified and fixed:**
1. ✅ **Model-Label Mismatch**: Model had 6 classes but labels.txt had 20 - causing incorrect gesture predictions
2. ✅ **TensorFlow Compatibility**: Added legacy Keras support for model loading
3. ✅ **Environment Setup**: Added .env.example and made Supabase optional

**The app now works correctly!** See guides below for details.

---

## 📚 Documentation

This project now includes comprehensive documentation:

- **[HOW_TO_RUN.md](HOW_TO_RUN.md)** - 🚀 **START HERE** - Step-by-step guide to run locally
- **[COMPLETE_SETUP_GUIDE.md](COMPLETE_SETUP_GUIDE.md)** - 🔧 Detailed troubleshooting and explanations
- **[CODEBASE_EXPLANATION.md](CODEBASE_EXPLANATION.md)** - 📖 Complete code walkthrough and architecture
- **[SETUP_GOOGLE_DRIVE.md](SETUP_GOOGLE_DRIVE.md)** - ☁️ Optional: Google Drive upload setup
- **[SUPABASE_SETUP.md](SUPABASE_SETUP.md)** - 🗄️ Optional: Database configuration

---

## ⚡ Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/atharv404/ann-pyproj.git
cd ann-pyproj

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the application
python app.py

# 4. Open in browser
# http://127.0.0.1:5000
```

**That's it!** The app works without additional configuration (Supabase is optional).

For detailed instructions, see **[HOW_TO_RUN.md](HOW_TO_RUN.md)**

---

## Supported Gestures (20 total)

The model currently recognizes 20 Indian Sign Language gestures:
1. **Namaste** - Traditional Indian greeting
2. **Good Morning** - Morning greeting
3. **Where?** - Question gesture
4. **Sorry** - Apology gesture
5. **Thirsty** - Need water
6. **Eat** - Food/eating
7. **Thank You** - Expressing gratitude
8. **Yes** - Affirmation
9. **No** - Negation
10. **Please** - Polite request
11. **Help** - Request for assistance
12. **Good** - Positive feedback
13. **Bad** - Negative feedback
14. **Stop** - Halt action
15. **Go** - Movement direction
16. **Come** - Invitation to approach
17. **Sit** - Seating action
18. **Stand** - Standing action
19. **Hello** - General greeting
20. **Goodbye** - Farewell

## Features

- **Real-time ISL Detection**: Live webcam feed with instant sign language recognition
- **Hand Detection with Bounding Box**: Uses MediaPipe to detect hands and show a green rectangle around the detected hand area
- **Focused Gesture Recognition**: Processes only the hand region, improving accuracy and reducing false positives from background objects
- **Visual Feedback**: See exactly where the system is detecting your hand with real-time bounding boxes and landmark visualization
- **Text-to-Speech**: Convert detected signs to speech output
- **Auto-Detection Logging**: Automatically saves detections with >70% confidence to Supabase
- **Google Drive Upload**: Upload training datasets directly to Google Drive
- **Detection History**: View metrics and history of all detections
- **Modern UI**: Professional dashboard with smooth animations

## Prerequisites

- Python 3.8 or higher
- Webcam
- Google account (optional - for Drive upload feature)
- Supabase account (optional - for detection logging)

**Note**: The app works without Supabase or Google Drive configuration. These are optional features.

**New**: The system now uses MediaPipe for hand detection, which provides:
- Accurate hand tracking with visible bounding boxes
- Better gesture recognition by focusing only on hand regions
- Reduced false positives from background objects

## Installation

1. **Clone or download the project**

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**

Create a `.env` file in the project root:
```env
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
```

4. **Set up Supabase database**

Run the SQL in `supabase_setup.sql` in your Supabase SQL Editor to create the required table.

5. **Set up Google Drive (Optional)**

Follow instructions in `SETUP_GOOGLE_DRIVE.md` to enable dataset uploads.

## Project Structure

```
MUDRA ISP/
├── app.py                      # Flask backend
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables
├── credentials.json            # Google OAuth credentials (after setup)
├── model.savedmodel/           # TensorFlow model
├── labels.txt                  # Class labels
├── templates/
│   └── index.html             # Frontend HTML
├── static/
│   ├── style.css              # Custom styles
│   └── script.js              # Frontend JavaScript
├── supabase_setup.sql         # Database schema
├── SETUP_GOOGLE_DRIVE.md      # Google Drive setup guide
└── README.md                  # This file
```

## Usage

1. **Start the application**
```bash
python app.py
```

2. **Open your browser**
```
http://127.0.0.1:5000
```

3. **Use the dashboard**
   - **Detection & Output**: Start webcam detection, view real-time translations, and use text-to-speech
   - **Dataset Upload**: Upload training images/videos to Google Drive
   - **Results & Metrics**: View detection history and statistics

## Features Explained

### Detection & Output
- Click "Start Detection" to begin real-time ISL recognition
- Detected signs appear in the text output panel
- Use "Click to Speak" for text-to-speech output
- Detections with >70% confidence are automatically saved to database

### Dataset Upload
- Upload training datasets directly to Google Drive
- Pre-configured folder or specify custom folder ID
- Supports multiple file uploads

### Results & Metrics
- Total detections count
- Average confidence score
- Most frequently detected sign
- Recent detection history (last 50)
- Clear history option

## Configuration

### Change Detection Confidence Threshold
Edit `app.py` line 148:
```python
if class_name != last_saved_class and confidence > 70:  # Change 70 to your threshold
```

### Change Default Google Drive Folder
Edit `templates/index.html` line 74 to update the default folder URL.

## Troubleshooting

### Camera not working
- Ensure webcam is connected and not used by another application
- Check browser permissions for camera access

### Google Drive upload fails
- Verify `credentials.json` exists in project root
- Delete `token.pickle` and re-authenticate
- Ensure you're added as a test user in Google Cloud Console

### Database errors
- Verify Supabase credentials in `.env`
- Ensure `recent_detections` table exists
- Check Row Level Security policies are set correctly

## Technologies Used

- **Backend**: Flask, TensorFlow, OpenCV
- **Frontend**: HTML, Tailwind CSS, JavaScript
- **Database**: Supabase (PostgreSQL)
- **Cloud Storage**: Google Drive API
- **ML Model**: TensorFlow SavedModel format

## License

This project is for educational purposes.

## Credits

Developed as part of the MUDRA ISL Detection System project.
