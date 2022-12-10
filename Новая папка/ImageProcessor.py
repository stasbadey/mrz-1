import numpy as np
from PIL import Image

from colorama import init

from settings import *


class ImageProcessor:
    def __init__(self):
        init()
        self.rects = np.empty(shape=(1, 1))
        self.rect_width = 0
        self.rect_height = 0
        self.width = 0
        self.height = 0

    def img_to_array(self, image_path):

        image = Image.open(image_path)

        img_arr = np.array(image)

        self.width = int(len(img_arr[0]))
        self.height = int(len(img_arr))

        print("Opened image", image_path)
        self.rects = self.__rect_split(img_arr)

        self.rects = [[(2 * x / MAX_COLOR_VAL) - 1 for x in rect] for rect in self.rects]
        self.rects = np.array(self.rects)

        return self.rects

    def to_image(self, array):
        array = (array.astype(float) + 1) * MAX_COLOR_VAL / 2
        # fixing rounding error
        array = np.clip(array, 0, 255)
        array = array.tolist()

        image = self.__rect_desplit(array)
        print('The image was desplitted')
        image = Image.fromarray(np.asarray(image).astype('uint8'))
        image.save('out.bmp')
        print('The image was saved as \'out.bmp\'')

        return image

    def __rect_split(self, arr):

        self.rect_height = 4
        self.rect_width = 4

        arr = arr.tolist()

        list_ = []
        self.__create_rect()
        rect_count = int((self.height / self.rect_height) * (self.width / self.rect_width))
        for rect in range(rect_count):
            list_.append([])
        for row_index in range(self.height):
            for col_index in range(self.width):
                for elem in arr[row_index][col_index]:
                    try:
                        list_index = ((row_index // self.rect_height) * int(self.width / self.rect_width)) + (
                                    col_index // self.rect_width)
                        list_[int(list_index)].append(elem)
                    except Exception:
                        pass

        return list_

    def __rect_desplit(self, array):
        list_ = []
        for row in range(self.height):
            list_.append([])
        for row_index in range(len(array)):
            for col_index in range(0, len(array[0]), COLOR_CODING_POSITIONS):
                pixel = array[row_index][col_index: col_index + COLOR_CODING_POSITIONS]
                list_index = (col_index // (COLOR_CODING_POSITIONS * self.rect_width)) + (
                            row_index // (self.width / self.rect_width) * self.rect_height)
                list_[int(list_index)].append(pixel)

        return list_

    def __calc_rect_side(self, img_side_len, coeff, factor_side):
        index = 0
        side = 1
        while side < img_side_len * coeff and index < len(factor_side):
            side *= factor_side[index]
            index += 1
        return side

    def __create_rect(self):
        self.rect_height = 4
        self.rect_width = 4
