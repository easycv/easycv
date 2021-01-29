import cv2

import uuid
from pathlib import Path
import subprocess as sp
import multiprocessing as mp
from os.path import relpath
import os
import shutil


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
    def __init__(self, path, temporary=False):
        self.path = path
        self.temporary = temporary
        self._uuid = None

        cap = cv2.VideoCapture(self.path)
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()

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

    def _process_chunk(self, info):
        cache_folder = Path(__file__).parent.absolute() / "cache"
        cap = cv2.VideoCapture(self.path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, info["start"])

        pipe = sp.Popen(
            info["cmd"]
            + [
                str(
                    cache_folder
                    / "{}-{}-{}.mp4".format(info["name"], info["start"], info["end"])
                )
            ],
            stdin=sp.PIPE,
            stderr=sp.PIPE,
        )

        processed_frames = 0
        while processed_frames <= (info["end"] - info["start"]):
            _, frame = cap.read()

            if frame is None:
                break

            frame = info["transform"].apply(frame)
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            pipe.stdin.write(frame.tobytes())
            processed_frames += 1

        cap.release()
        pipe.communicate(b"q")
        pipe.kill()
        return None

    def apply(self, transform, num_processes=2, preset="medium", in_place=False):
        cache_folder = Path(__file__).parent.absolute() / "cache"
        cache_folder.mkdir(exist_ok=True)

        cap = cv2.VideoCapture(self.path)
        _, first_frame = cap.read()
        first_frame = transform.apply(first_frame)
        width = first_frame.shape[1]
        height = first_frame.shape[0]
        cap.release()

        if in_place:
            name = self._uuid
        else:
            name = str(uuid.uuid4())

        cmd = generate_ffmpeg_cmd(width, height, self.fps, preset)

        p = mp.Pool(num_processes)
        chunks = self._create_chunks(num_processes, self.total_frames)
        info = []
        for chunk in chunks:
            chunk_info = {
                "start": chunk[0],
                "end": chunk[1],
                "transform": transform,
                "name": name,
                "cmd": cmd,
            }
            info.append(chunk_info)
        p.map(self._process_chunk, info)

        transport_streams = [
            cache_folder / "{}-{}-{}.mp4".format(name, start, end)
            for start, end in chunks
        ]

        intermediate = str(cache_folder / "{}-intermediate.txt".format(name))
        with open(intermediate, "w") as f:
            for t in transport_streams:
                f.write("file {} \n".format(str(t)))

        file = str(cache_folder / name) + ".mp4"
        ffmpeg_joining_command = "ffmpeg -y -loglevel warning -f concat -safe 0 "
        ffmpeg_joining_command += "-i {} -c copy -preset {} {}".format(
            intermediate, preset, file
        )

        sp.Popen(ffmpeg_joining_command, shell=True).wait()

        for f in transport_streams:
            os.remove(f)

        os.remove(intermediate)
        p.close()
        p.join()

        if in_place:
            self.path = file
        else:
            return Video(file, temporary=True)

    def save(self, filename):
        if self.temporary:
            self.temporary = False
            os.rename(self.path, filename)
        else:
            try:
                shutil.copy2(self.path, filename)
            except shutil.SameFileError:
                pass
        self.path = filename

    def close(self):
        if self.temporary:
            os.remove(self.path)

    def _repr_html_(self):
        html = '<video width="{}" height="{}" controls>'.format(720, 480)
        html += '<source src="{}"></video>'.format(relpath(self.path, os.getcwd()))
        return html
