import cv2
import librosa
import torch
import numpy as np
import os
import tensorflow as tf
# from google.cloud import storage
from transformers import AutoTokenizer, AutoModel
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from moviepy.editor import VideoFileClip
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
tf.get_logger().setLevel('ERROR')


# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'buzzapp-860e0-2eec481a04e9.json'


def extract_visual_features(video, sample_rate=1):
    # Load a pre-trained VGG16 model without the top classification layer
    model = VGG16(weights='imagenet', include_top=False)

    frames = []

    for t in np.arange(0, video.duration, sample_rate):
        frame = video.get_frame(t)
        frame = cv2.resize(frame, (224, 224))
        frames.append(frame)

    frames = np.array(frames)
    frames = preprocess_input(frames)
    features = model.predict(frames)
    features = features.mean(axis=(1, 2))

    return features.mean(axis=0)


def extract_audio_features(video, n_mfcc=200):
    audio_samples = video.audio.to_soundarray()
    audio_samples = audio_samples.mean(axis=1)  # Convert to mono
    sample_rate = video.audio.fps

    mfcc = librosa.feature.mfcc(y=audio_samples, sr=sample_rate, n_mfcc=n_mfcc)
    mfcc_mean = mfcc.mean(axis=1)
    
    return mfcc_mean


def extract_textual_features(text):
    if not text:
        return np.zeros(768)
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModel.from_pretrained('bert-base-uncased')

    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    cls_token = outputs.last_hidden_state[:, 0, :]

    return np.array(cls_token.detach().cpu().tolist())


def extract_features(video, caption):
    visual_features = extract_visual_features(video)
    audio_features = extract_audio_features(video)
    textual_features = extract_textual_features(caption)

    # Ensure all feature arrays have the same number of dimensions
    visual_features = np.squeeze(visual_features)
    textual_features = np.squeeze(textual_features)
    audio_features = np.squeeze(audio_features)

    combined_features = np.concatenate((visual_features, textual_features, audio_features), axis=-1)

    # potentially have reduce dimensions to fixed length
    return combined_features


def download_video(video_id):
    # Set up the Google Cloud Storage client
    client = storage.Client()

    # Replace these with your own Firebase Storage bucket name and path
    bucket_name = 'buzzapp-860e0.appspot.com'
    file_path = f'videos/{video_id}'


    # Download the video file to the /tmp directory
    # destination_path = f'/tmp/{video_id}.mp4'
    current_working_directory = os.getcwd()
    destination_path = os.path.join(current_working_directory, 'Video0.mp4')
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(file_path)
    blob.download_to_filename(destination_path)

    return destination_path


def lambda_handler(event, context):
    video_path = download_video(event['video_id'])
    video_caption = event['video_caption']
    video = VideoFileClip(video_path)

    video_features = extract_features(video, video_caption)
    print(len(video_features))



# lambda_handler({
#     "video_id": "Video5",
#     "video_caption": "fun concert, had so much fun seeing matroda live"
#     }, None)