# Instructions to Push to Your GitHub Repository

To push this code to your own GitHub repository, follow these steps:

1. Create a new repository on GitHub (without initializing it with README, .gitignore, or license)

2. Copy the repository URL (HTTPS or SSH)

3. Run the following commands in your terminal:

```bash
# Navigate to the project directory
cd /path/to/whisper-audio-extractor

# Add the remote repository
git remote add origin YOUR_REPOSITORY_URL

# Push the code to GitHub
git push -u origin master
```

Replace `YOUR_REPOSITORY_URL` with the URL of your GitHub repository.

## Running the Application

After downloading the code:

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure FFmpeg is installed on your system

3. Run the application:
```bash
python app.py
```

   If port 52678 is already in use, you can specify a different port:
```bash
python app.py --port 8080
```

4. Open your browser and navigate to:
```
http://localhost:52678
```
   (or the port you specified)

5. If you encounter NumPy-related errors, install a compatible version:
```bash
pip install numpy<2.0.0
```

6. Upload an MP4 file, add metadata, and see the transcription results!