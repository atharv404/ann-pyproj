# MUDRA - Indian Sign Language Detection System

A real-time Indian Sign Language (ISL) detection system using TensorFlow and Flask with automatic detection logging and Google Drive integration.

## Features

- **Real-time ISL Detection**: Live webcam feed with instant sign language recognition
- **Text-to-Speech**: Convert detected signs to speech output
- **Auto-Detection Logging**: Automatically saves detections with >70% confidence to Supabase
- **Google Drive Upload**: Upload training datasets directly to Google Drive
- **Detection History**: View metrics and history of all detections
- **Modern UI**: Professional dashboard with smooth animations

## Prerequisites

- Python 3.8 or higher
- Webcam
- Google account (for Drive upload feature)
- Supabase account (for detection logging)

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
