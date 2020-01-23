from cv.transforms.base import Transform
from cv.transforms.filter import Correlate
from cv.transforms.kernels import smooth_gradient_kernel


def normalize(x):
    return (x-x.min())/(x.max()-x.min())


class Gradient(Transform):
    arguments = {'size': 3, 'operator': 'sobel', 'sigma': 3}

    def apply(self, image):
        kernel_x = smooth_gradient_kernel(size=self.arguments['size'], operator=self.arguments['size'],
                                        axis='x', sigma=self.arguments['sigma'])
        kernel_y = smooth_gradient_kernel(size=self.arguments['size'], operator=self.arguments['size'],
                                          axis='y', sigma=self.arguments['sigma'])
        y = Correlate(kernel=kernel_y).apply(image)
        x = Correlate(kernel=kernel_x).apply(image)

        gradient = (x**2 + y**2)**0.5

        return normalize(gradient)

