from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
import os
from werkzeug.utils import secure_filename
from main import main  # Import your video processing function
import threading
import time

app = Flask(__name__)

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

# Ensure the upload and processed folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Global variable to track processing status
processing_status = {
    'is_processing': False,
    'progress': 0,
    'processed_file': None
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_video(input_path, output_path, start_time, end_time):
    global processing_status
    processing_status['is_processing'] = True
    processing_status['progress'] = 0
    processing_status['processed_file'] = None

    try:
        # Process the video with the specified time range
        main(input_path, output_path, processing_status, end_time)
        processing_status['processed_file'] = os.path.basename(output_path)
    except Exception as e:
        print(f"Error during video processing: {e}")
        raise
    finally:
        processing_status['is_processing'] = False

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            # Save the uploaded file
            filename = secure_filename(file.filename)
            input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(input_path)
            
            # Get start and end times from the form
            start_time = float(request.form['start-time'])
            end_time = float(request.form['end-time'])

            # Start video processing in a separate thread
            output_filename = f"processed_{filename}"
            output_path = os.path.join(app.config['PROCESSED_FOLDER'], output_filename)
            threading.Thread(target=process_video, args=(input_path, output_path, start_time, end_time)).start()
            
            return jsonify({
                'uploaded_video': filename,
                'is_processing': True
            })
    
    return render_template('index.html')

@app.route('/progress')
def progress():
    global processing_status
    return jsonify(processing_status)

@app.route('/download/<filename>')
def download(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename, as_attachment=True)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/processed/<filename>')
def processed_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)