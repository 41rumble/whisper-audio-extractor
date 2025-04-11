# Whisper Audio Extractor

A simple web application that allows users to upload MP4 videos, extract the audio, and transcribe it using OpenAI's Whisper model.

## Features

- Upload MP4 video files
- Add metadata (name and description)
- Extract audio from video
- Transcribe audio to text using OpenAI's Whisper
- Multiple accuracy options (tiny, base, small, medium, large models)
- Audio preprocessing to improve transcription quality
- Speaker diarization (identify different speakers in the audio)
- Language selection for better accuracy
- Save transcriptions to text files
- Simple and responsive web interface

## Requirements

- Python 3.8+
- FFmpeg (required for audio extraction)
- NumPy < 2.0.0 (compatibility requirement)
- PyTorch
- HuggingFace account and token (only for speaker diarization)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/whisper-audio-extractor.git
   cd whisper-audio-extractor
   ```

2. Install FFmpeg (if not already installed):
   - **Ubuntu/Debian**: `sudo apt-get install ffmpeg`
   - **macOS**: `brew install ffmpeg`
   - **Windows**: Download from [FFmpeg website](https://ffmpeg.org/download.html)

3. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Start the server:
   ```
   python app.py
   ```
   
   If port 52678 is already in use, you can specify a different port:
   ```
   python app.py --port 8080
   ```

2. Open your web browser and navigate to:
   ```
   http://localhost:52678
   ```
   (or the port you specified)

3. Use the web interface to:
   - Enter a name and description for your video
   - Select an MP4 file to upload
   - Choose a Whisper model size (larger models are more accurate but slower)
   - Select a language (optional, improves accuracy if known)
   - Enable speaker diarization if needed (requires HuggingFace token)
   - Click "Upload & Process" to start the extraction and transcription

4. Wait for the processing to complete (this may take some time depending on the video length and model size)

5. View the transcription results on the page, including speaker identification if enabled

### Speaker Diarization

To use speaker diarization (identifying different speakers):

1. Create a HuggingFace account at https://huggingface.co/
2. Generate an access token at https://huggingface.co/settings/tokens
3. Check the "Enable Speaker Diarization" option in the web interface
4. Enter your HuggingFace token in the field that appears

### Improving Accuracy

For the best transcription results:

1. Choose a larger model size (medium or large) for more accurate transcriptions
2. Specify the language if you know it
3. Use high-quality audio input
4. For multi-speaker content, enable speaker diarization

## Notes

- The application uses the "base" Whisper model by default, which balances accuracy and speed
- Maximum upload size is set to 500MB
- Temporary files are automatically cleaned up after processing
- This application requires NumPy < 2.0.0 due to compatibility issues with PyTorch
- If you encounter NumPy-related errors, run: `pip install numpy<2.0.0`

## License

MIT