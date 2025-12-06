from pydub import AudioSegment

audio = AudioSegment.from_file("Emine.mp4")
audio.export("output.mp3", format="mp3", bitrate="192k")

# audio.export("output.wav", format="wav")
