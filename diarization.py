"""
Speaker diarization module with multiple backend options.
"""

import os
import numpy as np
import torch
from pydub import AudioSegment
import tempfile
import librosa
from scipy.spatial.distance import cdist
from scipy.signal import medfilt
import warnings

# Global variables for models
pyannote_pipeline = None
speechbrain_model = None
resemblyzer_encoder = None

def perform_diarization(audio_path, method="pyannote", huggingface_token=None, **kwargs):
    """
    Perform speaker diarization using the specified method.
    
    Args:
        audio_path: Path to the audio file
        method: Diarization method to use ('pyannote', 'speechbrain', or 'resemblyzer')
        huggingface_token: Token for HuggingFace (required for pyannote)
        **kwargs: Additional parameters for specific methods
        
    Returns:
        List of segments with speaker IDs and timestamps
    """
    # Import globals
    global pyannote_pipeline, speechbrain_model, resemblyzer_encoder
    
    # Check if the specified method is available
    if method == "pyannote":
        try:
            from pyannote.audio import Pipeline
            import huggingface_hub
        except ImportError:
            print("PyAnnote is not available. Please install pyannote.audio.")
            return None
        
        # Validate token for pyannote
        if not huggingface_token:
            print("HuggingFace token is required for pyannote diarization")
            return None
            
        try:
            # Login with token
            huggingface_hub.login(token=huggingface_token)
            
            # Initialize the pipeline if not already done
            if pyannote_pipeline is None:
                print("Loading pyannote diarization pipeline...")
                
                # Explicitly download the segmentation model first
                try:
                    print("Downloading segmentation model...")
                    from huggingface_hub import hf_hub_download
                    
                    # Download the model files
                    segmentation_checkpoint = hf_hub_download(
                        repo_id="pyannote/segmentation-3.0",
                        filename="pytorch_model.bin",
                        token=huggingface_token
                    )
                    segmentation_config = hf_hub_download(
                        repo_id="pyannote/segmentation-3.0",
                        filename="config.yaml",
                        token=huggingface_token
                    )
                    print(f"Successfully downloaded segmentation model files")
                except Exception as e:
                    print(f"Error downloading segmentation model: {str(e)}")
                    raise
                
                # Load the pipeline
                pyannote_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=huggingface_token
                )
                
                # Move to GPU if available
                if torch.cuda.is_available():
                    print("Moving pyannote pipeline to GPU...")
                    pyannote_pipeline.to(torch.device("cuda"))
                
                print("Pyannote pipeline loaded successfully")
            
            # Run diarization
            print("Running pyannote diarization...")
            diarization = pyannote_pipeline(audio_path)
            
            # Convert to segments
            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append({
                    "speaker": speaker,
                    "start": turn.start,
                    "end": turn.end
                })
            
            print(f"Pyannote extracted {len(segments)} raw segments")
            return segments
            
        except Exception as e:
            print(f"Error in pyannote diarization: {str(e)}")
            return None
            
    elif method == "speechbrain":
        try:
            import speechbrain as sb
            from speechbrain.pretrained import EncoderClassifier
        except ImportError:
            print("SpeechBrain is not available. Please install speechbrain.")
            return None
            
        if not os.path.exists(audio_path):
            print(f"Audio file not found: {audio_path}")
            return None
            
        try:
            print("Using SpeechBrain for diarization (no HuggingFace token needed)...")
            
            # Load the model if not already loaded
            if speechbrain_model is None:
                print("Loading SpeechBrain speaker embedding model...")
                # Create model directory if it doesn't exist
                os.makedirs("pretrained_models", exist_ok=True)
                
                try:
                    # Try to load the model
                    speechbrain_model = EncoderClassifier.from_hparams(
                        source="speechbrain/spkrec-ecapa-voxceleb",
                        savedir="pretrained_models/spkrec-ecapa-voxceleb"
                    )
                    print("SpeechBrain model loaded successfully")
                except Exception as e:
                    print(f"Error loading SpeechBrain model: {str(e)}")
                    print("Trying alternative download method...")
                    
                    # Try alternative method with run_opts
                    speechbrain_model = EncoderClassifier.from_hparams(
                        source="speechbrain/spkrec-ecapa-voxceleb",
                        savedir="pretrained_models/spkrec-ecapa-voxceleb",
                        run_opts={"device": "cpu"}
                    )
                    print("SpeechBrain model loaded with alternative method")
            
            # Parameters for segmentation
            segment_len = 3.0  # seconds per segment
            step_len = 1.0     # step size in seconds
            
            # Load audio
            print("Loading and segmenting audio...")
            signal, fs = librosa.load(audio_path, sr=16000, mono=True)
            duration = len(signal) / fs
            
            # Create segments
            segments = []
            for start in np.arange(0, duration - segment_len, step_len):
                end = start + segment_len
                segments.append({
                    "start": start,
                    "end": end,
                    "signal": signal[int(start * fs):int(end * fs)]
                })
            
            if len(segments) == 0:
                print("Audio too short for segmentation")
                return None
                
            # Extract embeddings
            print(f"Extracting embeddings for {len(segments)} segments...")
            embeddings = []
            for segment in segments:
                with torch.no_grad():
                    embedding = speechbrain_model.encode_batch(
                        torch.tensor(segment["signal"]).unsqueeze(0)
                    ).squeeze().cpu().numpy()
                    embeddings.append(embedding)
            
            # Cluster embeddings
            print("Clustering speakers...")
            max_speakers = kwargs.get("max_speakers", 5)
            
            # Determine number of speakers (simple approach)
            from sklearn.cluster import AgglomerativeClustering
            
            # Try different numbers of speakers and pick the best
            best_score = float('inf')
            best_labels = None
            best_n_clusters = 2
            
            # Distance matrix
            distance_matrix = cdist(embeddings, embeddings, metric='cosine')
            
            # Try different numbers of clusters
            for n_clusters in range(2, min(max_speakers + 1, len(segments) + 1)):
                clustering = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    affinity='precomputed',
                    linkage='average'
                )
                labels = clustering.fit_predict(distance_matrix)
                
                # Calculate silhouette score (simplified)
                score = np.mean(distance_matrix[np.arange(len(labels)), labels])
                
                if score < best_score:
                    best_score = score
                    best_labels = labels
                    best_n_clusters = n_clusters
            
            print(f"Identified {best_n_clusters} speakers")
            
            # Create diarization segments
            diar_segments = []
            for i, segment in enumerate(segments):
                speaker = f"SPEAKER_{best_labels[i]}"
                diar_segments.append({
                    "speaker": speaker,
                    "start": segment["start"],
                    "end": segment["end"]
                })
            
            # Smooth the diarization (merge adjacent segments with same speaker)
            smoothed_segments = []
            if diar_segments:
                current = diar_segments[0].copy()
                for segment in diar_segments[1:]:
                    if segment["speaker"] == current["speaker"] and segment["start"] <= current["end"] + 0.5:
                        # Extend current segment
                        current["end"] = segment["end"]
                    else:
                        # Add current segment and start a new one
                        smoothed_segments.append(current)
                        current = segment.copy()
                
                # Add the last segment
                smoothed_segments.append(current)
            
            print(f"SpeechBrain extracted {len(smoothed_segments)} segments after smoothing")
            return smoothed_segments
            
        except Exception as e:
            print(f"Error in SpeechBrain diarization: {str(e)}")
            return None
            
    elif method == "resemblyzer":
        try:
            from resemblyzer import VoiceEncoder, preprocess_wav
            import librosa
        except ImportError:
            print("Resemblyzer is not available. Please install resemblyzer and librosa.")
            return None
            
        try:
            print("Using Resemblyzer for diarization (no HuggingFace token needed)...")
            
            # Load the model if not already loaded
            if resemblyzer_encoder is None:
                print("Loading Resemblyzer voice encoder...")
                resemblyzer_encoder = VoiceEncoder()
                print("Resemblyzer model loaded")
            
            # Load and preprocess audio
            print("Loading and preprocessing audio...")
            wav, sr = librosa.load(audio_path, sr=16000, mono=True)
            wav = preprocess_wav(wav)
            
            # Parameters for segmentation
            segment_len = 3.0  # seconds
            step_len = 1.0     # seconds
            samples_per_segment = int(segment_len * sr)
            samples_per_step = int(step_len * sr)
            
            # Create segments
            segments = []
            for i in range(0, len(wav) - samples_per_segment, samples_per_step):
                segment_wav = wav[i:i + samples_per_segment]
                start_time = i / sr
                end_time = (i + samples_per_segment) / sr
                segments.append({
                    "start": start_time,
                    "end": end_time,
                    "wav": segment_wav
                })
            
            if len(segments) == 0:
                print("Audio too short for segmentation")
                return None
            
            # Extract embeddings
            print(f"Extracting embeddings for {len(segments)} segments...")
            embeddings = []
            for segment in segments:
                embedding = resemblyzer_encoder.embed_utterance(segment["wav"])
                embeddings.append(embedding)
            
            # Cluster embeddings
            print("Clustering speakers...")
            from sklearn.cluster import AgglomerativeClustering
            
            # Determine number of speakers
            max_speakers = kwargs.get("max_speakers", 5)
            
            # Distance matrix
            embeddings_array = np.array(embeddings)
            distance_matrix = cdist(embeddings_array, embeddings_array, metric='cosine')
            
            # Try different numbers of clusters
            best_score = float('inf')
            best_labels = None
            best_n_clusters = 2
            
            for n_clusters in range(2, min(max_speakers + 1, len(segments) + 1)):
                clustering = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    affinity='precomputed',
                    linkage='average'
                )
                labels = clustering.fit_predict(distance_matrix)
                
                # Calculate silhouette score (simplified)
                score = np.mean(distance_matrix[np.arange(len(labels)), labels])
                
                if score < best_score:
                    best_score = score
                    best_labels = labels
                    best_n_clusters = n_clusters
            
            print(f"Identified {best_n_clusters} speakers")
            
            # Create diarization segments
            diar_segments = []
            for i, segment in enumerate(segments):
                speaker = f"SPEAKER_{best_labels[i]}"
                diar_segments.append({
                    "speaker": speaker,
                    "start": segment["start"],
                    "end": segment["end"]
                })
            
            # Smooth the diarization
            smoothed_segments = []
            if diar_segments:
                current = diar_segments[0].copy()
                for segment in diar_segments[1:]:
                    if segment["speaker"] == current["speaker"] and segment["start"] <= current["end"] + 0.5:
                        # Extend current segment
                        current["end"] = segment["end"]
                    else:
                        # Add current segment and start a new one
                        smoothed_segments.append(current)
                        current = segment.copy()
                
                # Add the last segment
                smoothed_segments.append(current)
            
            print(f"Resemblyzer extracted {len(smoothed_segments)} segments after smoothing")
            return smoothed_segments
            
        except Exception as e:
            print(f"Error in Resemblyzer diarization: {str(e)}")
            return None
    
    else:
        print(f"Unknown diarization method: {method}")
        return None

def post_process_diarization(segments, max_speakers=5, min_segment_duration=1.0):
    """
    Post-process diarization segments to improve quality.
    
    Args:
        segments: List of diarization segments
        max_speakers: Maximum number of speakers to keep
        min_segment_duration: Minimum duration for a segment in seconds
        
    Returns:
        Processed segments
    """
    if not segments:
        return []
        
    # Sort segments by time
    segments.sort(key=lambda x: x["start"])
    
    # Merge very short segments with the same speaker
    merged_segments = []
    current_segment = segments[0].copy()
    
    for segment in segments[1:]:
        # If same speaker and close in time, merge
        if (segment["speaker"] == current_segment["speaker"] and 
            segment["start"] - current_segment["end"] < 0.5):  # 500ms gap tolerance
            current_segment["end"] = segment["end"]
        else:
            # Only keep segments that are long enough
            if current_segment["end"] - current_segment["start"] >= min_segment_duration:
                merged_segments.append(current_segment)
            current_segment = segment.copy()
    
    # Add the last segment if it's long enough
    if current_segment["end"] - current_segment["start"] >= min_segment_duration:
        merged_segments.append(current_segment)
    
    print(f"Reduced to {len(merged_segments)} merged speaker segments after filtering short segments")
    
    # Limit the number of unique speakers
    speakers = set(s["speaker"] for s in merged_segments)
    
    if len(speakers) > max_speakers:
        print(f"Warning: Found {len(speakers)} speakers, which seems high. Limiting to {max_speakers}.")
        
        # Count speaker durations
        speaker_durations = {}
        for s in merged_segments:
            speaker = s["speaker"]
            duration = s["end"] - s["start"]
            if speaker not in speaker_durations:
                speaker_durations[speaker] = 0
            speaker_durations[speaker] += duration
        
        # Keep only the top speakers by total duration
        top_speakers = sorted(speaker_durations.keys(), 
                             key=lambda x: speaker_durations[x], 
                             reverse=True)[:max_speakers]
        
        # Map minor speakers to the closest major speaker
        speaker_mapping = {}
        for speaker in speakers:
            if speaker in top_speakers:
                speaker_mapping[speaker] = speaker
            else:
                # Map to the first speaker as fallback
                speaker_mapping[speaker] = top_speakers[0]
        
        # Apply the mapping
        for segment in merged_segments:
            segment["speaker"] = speaker_mapping[segment["speaker"]]
        
        print(f"Mapped to {len(set(s['speaker'] for s in merged_segments))} unique speakers")
    
    return merged_segments