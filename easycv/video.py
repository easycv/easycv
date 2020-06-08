import cv2

import uuid
from pathlib import Path
import subprocess as sp
import multiprocessing as mp
from os.path import relpath
import os


def generate_ffmpeg_cmd(width, height, fps, preset):
    ffmpeg_bin = "ffmpeg"
    command = [
        ffmpeg_bin,
        "-y",
        "-loglevel",
        "warning",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        "{}x{}".format(width, height),
        "-pix_fmt",
        "bgr24",
        "-r",
        "%.02f" % fps,
        "-an",
        "-i",
        "-",
        "-vcodec",
        "libx264",
        "-preset",
        preset,
    ]

    if (width % 2 == 0) and (height % 2 == 0):
        command.extend(["-pix_fmt", "yuv420p"])

    return command


class Video:
    def __init__(self, path):
        self.path = path
        self.uuid = str(uuid.uuid4())

        cap = cv2.VideoCapture(self.path)
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()

        self._current_cmd = None
        self._current_transform = None

    @staticmethod
    def _create_chunks(n, total):
        chunk_size = total // n
        chunks = []
        previous = -1
        for chunk in range(n - 1):
            start = previous + 1
            end = start + chunk_size
            chunks.append([int(start), int(end)])
            previous = end
        chunks.append([int(previous + 1), int(total)])
        return chunks

    def _process_chunk(self, chunk):
        start, end = chunk
        cache_folder = Path(__file__).parent.absolute() / "cache"
        cap = cv2.VideoCapture(self.path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

        pipe = sp.Popen(
            self._current_cmd
            + [str(cache_folder / "{}-{}-{}.mp4".format(self.uuid, start, end))],
            stdin=sp.PIPE,
            stderr=sp.PIPE,
        )

        processed_frames = 0
        while processed_frames <= (end - start):
            _, frame = cap.read()

            if frame is None:
                break

            frame = self._current_transform.apply(frame)
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            # frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2YUV_I420)
            pipe.stdin.write(frame.tobytes())
            processed_frames += 1

        cap.release()
        pipe.communicate(b"q")
        pipe.kill()
        return None

    def apply(self, transform, num_processes=2, preset="medium"):
        cache_folder = Path(__file__).parent.absolute() / "cache"
        cache_folder.mkdir(exist_ok=True)

        cap = cv2.VideoCapture(self.path)
        _, first_frame = cap.read()
        first_frame = transform.apply(first_frame)
        width = first_frame.shape[1]
        height = first_frame.shape[0]
        cap.release()

        self._current_cmd = generate_ffmpeg_cmd(width, height, self.fps, preset)
        self._current_transform = transform

        p = mp.Pool(num_processes)
        chunks = self._create_chunks(num_processes, self.total_frames)
        p.map(self._process_chunk, chunks)

        transport_streams = [
            cache_folder / "{}-{}-{}.mp4".format(self.uuid, start, end)
            for start, end in chunks
        ]

        intermediate = str(cache_folder / "{}-intermediate.txt".format(self.uuid))
        with open(intermediate, "w") as f:
            for t in transport_streams:
                f.write("file {} \n".format(str(t)))

        file = str(cache_folder / self.uuid) + ".mp4"
        ffmpeg_joining_command = "ffmpeg -y -loglevel warning -f concat -safe"
        ffmpeg_joining_command += "0 -i {} -c copy -preset {} {}".format(
            intermediate, preset, file
        )

        sp.Popen(ffmpeg_joining_command, shell=True).wait()

        from os import remove

        for f in transport_streams:
            remove(f)

        remove(intermediate)
        p.close()
        p.join()

        return Video(file)

    def pau(self):
        html = '<video width="{}" height="{}" controls>'.format(720, 480)
        html += '<source src="{}"></video>'.format(relpath(self.path, os.getcwd()))
        return html

    def _repr_html_(self):
        html = '<video width="{}" height="{}" controls>'.format(720, 480)
        html += '<source src="{}"></video>'.format(relpath(self.path, os.getcwd()))
        return html
