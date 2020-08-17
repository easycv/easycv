import cv2

from easycv.pipeline import Pipeline
from easycv.image import Image


class Camera:
    def __init__(self, device=0):
        self._device = device
        self._pending = Pipeline([])

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

    def apply(self, operation):
        self._pending.add_transform(operation)

    def clear(self):
        self._pending.clear()
