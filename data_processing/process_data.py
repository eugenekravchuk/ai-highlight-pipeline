from pydub import AudioSegment

audio = AudioSegment.from_file("england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley/1_720p.mkv")
audio.export("output.mp3", format="mp3", bitrate="192k")

audio.export("output.wav", format="wav")
