from ImageProcessor import ImageProcessor
from NeuralCompressor import NeuralCompressor

if __name__ == "__main__":
    image_processor = ImageProcessor()
    prepared_image = image_processor.img_to_array('image/1.bmp')

    compressed_matrix = NeuralCompressor(
        p=32,
        err=20000,
        a=0.0001,
        prepared_image=prepared_image
    ).compress()

    image_processor.to_image(compressed_matrix)
