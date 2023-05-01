from moviepy.editor import VideoFileClip

video_path = "data/videos/{}.mov"

class VideoPost:

	def __init__(self, video_path, caption, location):
		self.video = VideoFileClip(video_path)
		self.title = video_path
		self.caption = caption
		self.location = location


video1 = VideoPost(
	video_path.format("chris_lake_fisher_coachella_1"),
	"Chris Lake and Fisher drop a banger at Coachella",
	"Coachella"
	)


video2 = VideoPost(
	video_path.format("chris_lake_fisher_coachella_2"),
	"House music was served at Coachella this weekend by Chris Lake and Fisher",
	"Coachella"
	)

video3 = VideoPost(
	video_path.format("fourtet_dubstep_1"),
	"Fourtet is crazy bro",
	""
	)

video4 = VideoPost(
	video_path.format("skrillex_fred_again_fourtet_coachella_1"),
	"Skrillex, Fourtet, and Fred Again just absolutely blew up Coachella",
	"Coachella"
	)

video5 = VideoPost(
	video_path.format("subtronics_visuals_dubstep"),
	"Visuals I did for subtronics",
	""
	)