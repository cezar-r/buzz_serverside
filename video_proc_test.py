import numpy as np

from video_proc import extract_visual_features, extract_audio_features, extract_textual_features
from video_proc_data import *


def cosine_sim(vec1, vec2):
	vec2 = vec2.T
	dot_product = np.dot(vec1, vec2)

	magnitude_vec1 = np.linalg.norm(vec1)
	magnitude_vec2 = np.linalg.norm(vec2)

	return dot_product / (magnitude_vec1 * magnitude_vec2)


videos = [video1, video2, video3, video4, video5]
video_data = {}

for video in videos:
	print(f"Getting features for {video.title}")
	this_video_data = {}

	this_video_data['visual'] = extract_visual_features(video.video)
	this_video_data['audio'] = extract_audio_features(video.video)
	this_video_data['caption'] = extract_textual_features(video.caption)
	this_video_data['location'] = extract_textual_features(video.location)
	this_video_data['text'] = extract_textual_features(video.caption + video.location)

	video_data[video.title] = this_video_data
print()

for video in video_data:
	print(f"Looking for most similar video to {video}")

	this_data = video_data[video]

	max_visual_sim = 0
	max_visual_sim_vid = ""

	max_audio_sim = 0
	max_audio_sim_vid = ""

	max_caption_sim = 0
	max_caption_sim_vid = ""

	max_location_sim = 0
	max_location_sim_vid = ""

	for other_video in video_data:
		if other_video != video:
			other_data = video_data[other_video]
			visual_sim = cosine_sim(this_data['visual'], other_data['visual'])
			if visual_sim > max_visual_sim:
				max_visual_sim = visual_sim
				max_visual_sim_vid = other_video

			audio_sim = cosine_sim(this_data['audio'], other_data['audio'])
			if audio_sim > max_audio_sim:
				max_audio_sim = audio_sim
				max_audio_sim_vid = other_video

			caption_sim = cosine_sim(this_data['caption'], other_data['caption'])
			if caption_sim > max_caption_sim:
				max_caption_sim = caption_sim
				max_caption_sim_vid = other_video

			location_sim = cosine_sim(this_data['location'], other_data['location'])
			if location_sim > max_location_sim:
				max_location_sim = location_sim
				max_location_sim_vid = other_video

	best_videos = np.array([max_visual_sim_vid, max_audio_sim_vid, max_caption_sim_vid, max_location_sim_vid])
	unique, counts = np.unique(best_videos, return_counts=True)
	mode = unique[counts.argmax()]

	print(f"Most similar visual features: {max_visual_sim_vid} ({max_visual_sim})")
	print(f"Most similar audio features: {max_audio_sim_vid} ({max_audio_sim})")
	print(f"Most similar caption features: {max_caption_sim_vid} ({max_caption_sim})")
	print(f"Most similar location features: {max_location_sim_vid} ({max_location_sim})")
	print(f"Recommended video: {mode}")
	print("\n")