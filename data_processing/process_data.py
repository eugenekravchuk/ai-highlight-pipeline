from pydub import AudioSegment

# Потрібен ffmpeg у PATH
audio = AudioSegment.from_file("england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley/1_720p.mkv")   # автоматично знайде аудіопотік
audio.export("output.mp3", format="mp3", bitrate="192k")
# або
audio.export("output.wav", format="wav")
