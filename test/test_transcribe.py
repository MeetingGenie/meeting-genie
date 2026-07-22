from meeting_genie.audio import AudioRecorder
from meeting_genie.transcribe import run_transcription_loop
import yaml

cfg = yaml.safe_load(open("meeting_genie/config.yaml"))

recorder = AudioRecorder()
recorder.start()

def on_utterance(u):
    print(u)

run_transcription_loop(recorder, cfg, on_utterance)
com