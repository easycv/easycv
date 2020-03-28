import numpy as np

from easycv.transforms.base import Transform


class MagnitudeSpectrum(Transform):
    default_args = {"scale": 15}

    def apply(self, image, **kwargs):
        fourier = np.fft.fft2(image)
        fourier_shifted = np.fft.fftshift(fourier)
        magnitude = kwargs["scale"] * np.log(1 + np.abs(fourier_shifted))
        return magnitude


class PhaseSpectrum(Transform):
    def apply(self, image, **kwargs):
        fourier = np.fft.fft2(image)
        fourier_shifted = np.fft.fftshift(fourier)
        phase = (np.angle(fourier_shifted) + np.pi) / (2 * np.pi)
        return phase * 255
