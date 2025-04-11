import os
import sys
import tempfile
import uuid
import json
import time
import re

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
from pydub import AudioSegment
import torch

# Import pyannote.audio for speaker diarization if available
DIARIZATION_AVAILABLE = False
try:
    from pyannote.audio import Pipeline
    DIARIZATION_AVAILABLE = True
    print("Speaker diarization is available")
except ImportError:
    print("Speaker diarization is not available. Install pyannote.audio for speaker identification.")
    
# HuggingFace token for pyannote.audio (needed for speaker diarization)
# Users will need to provide their own token
HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACE_TOKEN", None)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max upload size
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load Whisper model (small model for faster processing)
model = None
diarization_pipeline = None

def preprocess_audio(audio_path, output_path=None):
    """
    Preprocess audio to improve transcription quality.
    - Normalize audio
    - Remove background noise (simple method)
    - Convert to WAV format with appropriate parameters
    """
    if output_path is None:
        output_path = audio_path
    
    try:
        # Load audio
        audio = AudioSegment.from_file(audio_path)
        
        # Normalize audio (adjust volume to a standard level)
        normalized_audio = audio.normalize()
        
        # Export with optimal parameters for speech recognition
        normalized_audio.export(
            output_path,
            format="wav",
            parameters=["-ac", "1", "-ar", "16000"]  # Mono, 16kHz
        )
        
        print(f"Audio preprocessed and saved to {output_path}")
        return output_path
    except Exception as e:
        print(f"Error preprocessing audio: {str(e)}")
        return audio_path  # Return original path if preprocessing fails

def perform_diarization(audio_path, huggingface_token=None):
    """
    Perform speaker diarization on the audio file.
    Returns a list of segments with speaker IDs and timestamps.
    """
    global diarization_pipeline
    
    if not DIARIZATION_AVAILABLE:
        print("Speaker diarization is not available")
        return None
    
    if huggingface_token is None:
        print("No HuggingFace token provided for speaker diarization")
        return None
    
    try:
        # Initialize the diarization pipeline if not already done
        if diarization_pipeline is None:
            print("Loading speaker diarization model...")
            diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=huggingface_token
            )
        
        # Run the diarization
        print("Performing speaker diarization...")
        diarization = diarization_pipeline(audio_path)
        
        # Convert to a more usable format
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "speaker": speaker,
                "start": turn.start,
                "end": turn.end
            })
        
        return segments
    except Exception as e:
        print(f"Error during speaker diarization: {str(e)}")
        return None

def merge_diarization_with_transcription(transcription, diarization_segments):
    """
    Merge the diarization information with the transcription.
    This is a simple approach that assigns speakers based on timestamp overlap.
    """
    if not diarization_segments or not transcription.get("segments"):
        return transcription
    
    result = transcription.copy()
    
    # For each transcription segment, find the most overlapping speaker
    for segment in result["segments"]:
        segment_start = segment["start"]
        segment_end = segment["end"]
        
        # Find the speaker with the most overlap
        max_overlap = 0
        assigned_speaker = None
        
        for diar_segment in diarization_segments:
            diar_start = diar_segment["start"]
            diar_end = diar_segment["end"]
            
            # Calculate overlap
            overlap_start = max(segment_start, diar_start)
            overlap_end = min(segment_end, diar_end)
            overlap = max(0, overlap_end - overlap_start)
            
            if overlap > max_overlap:
                max_overlap = overlap
                assigned_speaker = diar_segment["speaker"]
        
        # Assign the speaker to the segment
        if assigned_speaker:
            segment["speaker"] = assigned_speaker
    
    return result

def format_transcript_with_speakers(transcription):
    """
    Format the transcription with speaker labels.
    """
    if not transcription.get("segments") or not any("speaker" in segment for segment in transcription["segments"]):
        return transcription["text"]
    
    formatted_text = []
    current_speaker = None
    current_text = []
    
    for segment in transcription["segments"]:
        speaker = segment.get("speaker", "Unknown")
        text = segment["text"].strip()
        
        if speaker != current_speaker:
            # Save the previous speaker's text
            if current_text:
                formatted_text.append(f"{current_speaker}: {' '.join(current_text)}")
                current_text = []
            
            current_speaker = speaker
        
        current_text.append(text)
    
    # Add the last speaker's text
    if current_text:
        formatted_text.append(f"{current_speaker}: {' '.join(current_text)}")
    
    return "\n\n".join(formatted_text)

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
    
    # Get metadata and options
    name = request.form.get('name', 'Unnamed')
    description = request.form.get('description', '')
    model_size = request.form.get('model_size', 'base')  # tiny, base, small, medium, large
    language = request.form.get('language', None)  # Language code or None for auto-detection
    enable_diarization = request.form.get('enable_diarization', 'false').lower() == 'true'
    huggingface_token = request.form.get('huggingface_token', HUGGINGFACE_TOKEN)
    
    # Log metadata
    print(f"Received file: {file.filename}")
    print(f"Metadata - Name: {name}, Description: {description}")
    print(f"Options - Model: {model_size}, Language: {language}, Diarization: {enable_diarization}")
    
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
            
            # Preprocess audio to improve quality
            print("Preprocessing audio...")
            processed_audio_path = audio_path + "_processed.wav"
            processed_audio_path = preprocess_audio(audio_path, processed_audio_path)
            
            # Perform speaker diarization if requested and available
            diarization_segments = None
            if enable_diarization and DIARIZATION_AVAILABLE and huggingface_token:
                print("Performing speaker diarization...")
                diarization_segments = perform_diarization(processed_audio_path, huggingface_token)
            
            # Load model if not already loaded or if a different model size is requested
            if model is None or (hasattr(model, 'model_size') and model.model_size != model_size):
                print(f"Loading Whisper {model_size} model...")
                model = whisper.load_model(model_size)
            
            # Prepare transcription options
            transcribe_options = {
                "fp16": torch.cuda.is_available(),  # Use FP16 if CUDA is available
                "temperature": 0,  # Lower temperature for more deterministic outputs
                "verbose": True,   # Print progress
            }
            
            # Add language if specified
            if language:
                transcribe_options["language"] = language
            
            # Transcribe audio with improved options
            print("Transcribing audio...")
            result = model.transcribe(processed_audio_path, **transcribe_options)
            
            # Merge diarization with transcription if available
            if diarization_segments:
                result = merge_diarization_with_transcription(result, diarization_segments)
                formatted_text = format_transcript_with_speakers(result)
            else:
                formatted_text = result["text"]
            
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
                f.write(f"Model: {model_size}\n")
                if language:
                    f.write(f"Language: {language}\n")
                f.write(f"Speaker diarization: {'Enabled' if diarization_segments else 'Disabled'}\n")
                f.write("\n--- TRANSCRIPTION ---\n\n")
                f.write(formatted_text)
                
                # Save raw JSON data for further processing if needed
                f.write("\n\n--- RAW DATA ---\n")
                f.write(json.dumps(result, indent=2))
            
            print(f"Transcription saved to: {transcription_path}")
            print("\n--- TRANSCRIPTION ---\n")
            print(formatted_text)
            print("\n--- END TRANSCRIPTION ---\n")
            
            # Clean up temporary files
            os.unlink(audio_path)
            if os.path.exists(processed_audio_path) and processed_audio_path != audio_path:
                os.unlink(processed_audio_path)
            os.unlink(video_path)
            
            return jsonify({
                'success': True,
                'transcription': formatted_text,
                'transcription_file': transcription_path,
                'has_speaker_diarization': diarization_segments is not None,
                'metadata': {
                    'name': name,
                    'description': description,
                    'model': model_size,
                    'language': language
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