import os
import sys
import tempfile
import uuid

# Check NumPy version before importing other dependencies
try:
    import numpy as np
    numpy_version = np.__version__
    major_version = int(numpy_version.split('.')[0])
    if major_version >= 2:
        print("\n" + "="*80)
        print("ERROR: Incompatible NumPy version detected:", numpy_version)
        print("This application requires NumPy < 2.0 due to compatibility issues with PyTorch.")
        print("\nPlease downgrade NumPy by running:")
        print("    pip uninstall -y numpy")
        print("    pip install numpy<2.0.0")
        print("="*80 + "\n")
        sys.exit(1)
except ImportError:
    # NumPy not installed yet, which is fine
    pass

from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from moviepy.editor import VideoFileClip
import whisper

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max upload size
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load Whisper model (small model for faster processing)
model = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global model
    
    # Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Get metadata
    name = request.form.get('name', 'Unnamed')
    description = request.form.get('description', '')
    
    # Log metadata
    print(f"Received file: {file.filename}")
    print(f"Metadata - Name: {name}, Description: {description}")
    
    if file:
        # Generate a unique filename
        filename = secure_filename(file.filename)
        unique_id = str(uuid.uuid4())
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{unique_id}_{filename}")
        file.save(video_path)
        
        try:
            # Extract audio from video
            print(f"Extracting audio from {video_path}...")
            video = VideoFileClip(video_path)
            
            # Create a temporary file for the audio
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                audio_path = temp_audio.name
            
            # Extract audio to the temporary file
            video.audio.write_audiofile(audio_path, codec='pcm_s16le')
            video.close()
            
            # Load model if not already loaded
            if model is None:
                print("Loading Whisper model...")
                model = whisper.load_model("base")
            
            # Transcribe audio
            print("Transcribing audio...")
            result = model.transcribe(audio_path)
            transcription = result["text"]
            
            # Create a results directory if it doesn't exist
            results_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'transcriptions')
            os.makedirs(results_dir, exist_ok=True)
            
            # Save the transcription to a file
            original_filename = os.path.splitext(os.path.basename(file.filename))[0]
            timestamp = uuid.uuid4().hex[:8]
            transcription_filename = f"{original_filename}_{timestamp}.txt"
            transcription_path = os.path.join(results_dir, transcription_filename)
            
            with open(transcription_path, 'w', encoding='utf-8') as f:
                f.write(f"Name: {name}\n")
                f.write(f"Description: {description}\n")
                f.write(f"Original file: {file.filename}\n")
                f.write("\n--- TRANSCRIPTION ---\n\n")
                f.write(transcription)
            
            print(f"Transcription saved to: {transcription_path}")
            print("\n--- TRANSCRIPTION ---\n")
            print(transcription)
            print("\n--- END TRANSCRIPTION ---\n")
            
            # Clean up temporary files
            os.unlink(audio_path)
            os.unlink(video_path)
            
            return jsonify({
                'success': True,
                'transcription': transcription,
                'transcription_file': transcription_path,
                'metadata': {
                    'name': name,
                    'description': description
                }
            })
            
        except Exception as e:
            # Clean up in case of error
            if os.path.exists(video_path):
                os.unlink(video_path)
            if 'audio_path' in locals() and os.path.exists(audio_path):
                os.unlink(audio_path)
            
            print(f"Error processing file: {str(e)}")
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Unknown error'}), 500

if __name__ == '__main__':
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='MP4 Audio Extractor & Transcriber')
    parser.add_argument('--port', type=int, default=52678, help='Port to run the server on (default: 52678)')
    args = parser.parse_args()
    
    # Run the app with the specified port
    try:
        app.run(host='0.0.0.0', port=args.port, debug=True)
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"\nERROR: Port {args.port} is already in use.")
            print(f"Please try a different port with: python app.py --port <port_number>\n")
        else:
            raise