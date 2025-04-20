import cv2
import imageio
import numpy as np
from pathlib import Path
import os
class VideoRecorder:
    def __init__(self, root_dir, render_size=84, fps=30, name="third"):
        self.name = name
        if root_dir is not None:
            self.save_dir = root_dir + '/eval_video' + "/{}/".format(name)
            self.save_dir = Path(self.save_dir)

            os.makedirs(self.save_dir, exist_ok=True)
        else:
            self.save_dir = None

        self.render_size = render_size
        self.fps = fps
        self.frames = []

    def init(self,  enabled=True):
        self.frames = []
        self.enabled = self.save_dir is not None and enabled


    def record(self, frame):
            self.frames.append(frame)

    def save(self, file_name):
        if self.enabled:
            path = self.save_dir / file_name
            imageio.mimsave(str(path), self.frames, fps=self.fps)
            self.init()