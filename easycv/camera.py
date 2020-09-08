import cv2

import os
import time
import uuid
from pathlib import Path

from easycv.pipeline import Pipeline
from easycv.video import Video
from easycv.image import Image


class Camera:
    def __init__(self, device=0):
        self._device = device
        self._pending = Pipeline([])
        self._fps = None

    def fps(self, frames="auto"):
        if self._fps is None:
            cap = cv2.VideoCapture(self._device)

            if frames == "auto":
                reference_fps = cap.get(cv2.CAP_PROP_FPS)
                frames = int(reference_fps * 5)

            start = time.time()
            for i in range(0, frames):
                cap.read()
            end = time.time()
            cap.release()
            self._fps = frames / (end - start)
        return self._fps

    def show(self, name="Camera"):
        cap = cv2.VideoCapture(self._device)
        ret, frame = cap.read()
        cv2.namedWindow(name, cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(name, frame.shape[1], frame.shape[0])

        while cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) >= 1:
            cv2.imshow(name, Image(frame).apply(self._pending).array)
            ret, frame = cap.read()
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                cv2.destroyAllWindows()
                break

        cap.release()
        cv2.destroyAllWindows()

    def capture(self):
        cap = cv2.VideoCapture(self._device)
        ret, frame = cap.read()
        cap.release()
        return Image(frame).apply(self._pending)

    def record(self, duration, preview=False, name="Preview"):
        cache_folder = Path(__file__).parent.absolute() / "cache"
        cache_folder.mkdir(exist_ok=True)
        temp_file = str(cache_folder / (str(uuid.uuid4()) + ".mp4"))
        filepath = str(cache_folder / (str(uuid.uuid4()) + ".mp4"))

        fps = self.fps()
        cap = cv2.VideoCapture(self._device)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc("M", "P", "4", "V")
        out = cv2.VideoWriter(temp_file, fourcc, fps, (frame_width, frame_height))

        if preview:
            cv2.namedWindow(name, cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow(name, frame_width, frame_height)

        for i in range(int(fps * duration) + 1):
            ret, frame = cap.read()
            frame = Image(frame).apply(self._pending).array

            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            if preview:
                cv2.imshow(name, frame)

            out.write(frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        os.system("ffmpeg -i {} -vcodec libx264 {}".format(temp_file, filepath))

        return Video(filepath, temporary=True)

    def apply(self, operation):
        self._pending.add_transform(operation)

    def pending(self):
        return self._pending

    def clear(self):
        self._pending.clear()
