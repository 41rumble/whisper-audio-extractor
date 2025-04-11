# Whisper Audio Extractor

A simple web application that allows users to upload MP4 videos, extract the audio, and transcribe it using OpenAI's Whisper model.

## Features

- Upload MP4 video files
- Add metadata (name and description)
- Extract audio from video
- Transcribe audio to text using Whisper
- Save transcriptions to text files
- Simple and responsive web interface

## Requirements

- Python 3.8+
- FFmpeg (required for audio extraction)

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
   - Click "Upload & Process" to start the extraction and transcription

4. Wait for the processing to complete (this may take some time depending on the video length)

5. View the transcription results on the page

## Notes

- The application uses the "base" Whisper model by default, which balances accuracy and speed
- Maximum upload size is set to 500MB
- Temporary files are automatically cleaned up after processing
- This application requires NumPy < 2.0.0 due to compatibility issues with PyTorch
- If you encounter NumPy-related errors, run: `pip install numpy<2.0.0`

## License

MIT