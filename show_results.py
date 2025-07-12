# Módulo para mostrar resultados dos modelos

from IPython.display import HTML
from base64 import b64encode
import os
import moviepy.editor as moviepy
from pathlib import Path


ROOT = Path(__file__).parent
COMPILED_FILES = ROOT / "compiled_files/object_counting_output.avi"
VIDEO_RESULT = ROOT / "result_files"


def exibir_video(video_caminho):

  mp4 = open(video_caminho, "rb").read()
  data_url = "data:video/mp4;base64," + b64encode(mp4).decode()

  return HTML("""
  <video width=500 controls>
    <source src='%s' type='video/mp4'>
    </video>
    """ % data_url)

video_salvo = str(COMPILED_FILES)
video_final = str(VIDEO_RESULT / "video_countingCar_result01.mp4")

clip = moviepy.VideoFileClip(video_salvo)
clip.write_videofile(video_final)


exibir_video(video_final)