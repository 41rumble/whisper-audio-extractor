import os
import tempfile
import uuid
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
            
            # Clean up temporary files
            os.unlink(audio_path)
            os.unlink(video_path)
            
            return jsonify({
                'success': True,
                'transcription': transcription,
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
    app.run(host='0.0.0.0', port=52678, debug=True)