import numpy as np
from time import time

from colorama import Fore


class NeuralCompressor:
    def __init__(self, p, a, err, prepared_image):
        self.p = p
        self.a = a
        self.err = err
        self.prepared_image = prepared_image

        self.N = len(self.prepared_image[0])
        self.L = self.prepared_image.shape[0] * self.prepared_image.shape[1]
        self.Z = (self.N * self.L) / ((self.N + self.L) * self.p + 2)
        print(Fore.BLUE, 'Z = ', self.Z)

    def compress(self):
        w1 = np.random.rand(self.N, self.p) * 2 - 1
        w2 = np.array(w1.transpose())
        step = 0
        time_ = 0.

        while True:
            error_common = 0
            step += 1

            time_start = time()

            for row in self.prepared_image:
                row = row.reshape(1, -1)
                # Y(i) = X(i)*W
                y = np.matmul(row, w1)
                # X'(i) = Y(i)*W'
                x = np.matmul(y, w2)
                # ∆X(i) = X'(i) – X(i)
                dx = x - row
                dx = dx.reshape(1, -1)

                # W2(t + 1) = W(t) - α*[Y(i) T * ∆X(i)]
                w2 -= self.a * np.matmul(y.transpose(), dx)
                # W1(t + 1) = W(t) - α*[X(i)] T *∆X(i)*[W'(t)]^T
                w1 -= self.a * np.matmul(np.matmul(row.transpose(), dx), w2.transpose())

                # Е(q) = ∑∆X(q)i *∆X(q)i
                error = (dx * dx).sum()
                error_common += error

            time_finish = time()
            time_ += time_finish - time_start
            print(Fore.YELLOW, 'Iteration', step, 'Time:', time_, 'sec., ',
                  'E:', error_common)

            if error_common < self.err:
                compressed_matrix = []
                for row in self.prepared_image:
                    compressed_matrix.append(row)
                return compressed_matrix
