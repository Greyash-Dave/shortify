# Shortify - Video to Short Converter 🎥✂️

Convert long videos into engaging short clips with auto-cropping, captions, and processing!

<a href="https://youtu.be/NynlLpj6_SE">
  <img src="https://img.youtube.com/vi/NynlLpj6_SE/0.jpg" alt="Watch the video" width="100%">
</a> 

🔗 [Watch the video on YouTube](https://www.youtube.com/watch?v=NynlLpj6_SE) 

# <p align="center">
#   <img src="https://raw.githubusercontent.com/Greyash-Dave/Greyash-Dave/main/images/shortify/1.PNG" alt="Screen Shot Image">
# </p>

<p align="center">
  <img src="https://raw.githubusercontent.com/Greyash-Dave/Greyash-Dave/main/images/shortify/2.PNG" alt="Screen Shot Image">
</p>

## Features ✨
- 🎞️ Upload videos (MP4, AVI, MOV, MKV)
- ⏱️ Set custom start/end times
- 🤖 Automatic face/human detection & tracking
- 📝 AI-generated captions using Whisper
- 🖼️ Auto-crop to 9:16 aspect ratio
- 📈 Real-time progress tracking
- 🌙 Dark/Light mode toggle

## Prerequisites 📋
- Python 3.11+
- FFmpeg
- Whisper model base (automatically downloaded)

## Installation ⚙️

### 1. Clone Repository
```bash
git clone https://github.com/Greyash-Dave/shortify.git
cd shortify
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Install FFmpeg
**Windows:**
```bash
choco install ffmpeg
```

**Mac:**
```bash
brew install ffmpeg
```

**Linux:**
```bash
sudo apt install ffmpeg
```

## Configuration ⚙️
1. Create required directories:
```bash
mkdir uploads processed
```

2. Set FFmpeg path in `main.py`:
```python
change_settings({"FFMPEG_BINARY": "ffmpeg"})  # Update path if needed
```

## Usage 🚀
### Local Development
```bash
python app.py
```

Visit http://localhost:5000

### Basic Workflow
1. Upload a video file
2. Set end time (0-60 seconds)
3. Watch real-time processing
4. Download processed short

## Deployment ☁️
### Vercel (Limited)
```bash
vercel
```

### Recommended Hosting (Fly.io)
1. Create `Dockerfile`:
```dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y ffmpeg

WORKDIR /app
COPY . .
RUN pip install -r requirements.txt

CMD gunicorn --bind 0.0.0.0:$PORT app:app
```

2. Deploy:
```bash
flyctl launch
```

## Troubleshooting 🔧
### Common Issues
**FFmpeg Errors:**
* Verify FFmpeg installation: `ffmpeg -version`
* Update path in `main.py`

**Large File Processing:**
* Keep videos under 60 seconds
* Use recommended hosting solutions

**Dependency Issues:**
```bash
pip freeze > requirements.txt  # Update if adding new packages
```

## Contributing 🤝
1. Fork the repository
2. Create your feature branch
3. Commit changes
4. Push to the branch
5. Open a Pull Request


**Happy Video Processing!** 🎬 If you enjoy this project, please ⭐ the repository!
