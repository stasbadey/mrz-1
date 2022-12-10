import os
import numpy as np


def matrix_mul(matrix1, matrix2):
    """
    Matrix multiplicity.
    :param matrix1: First plurality.
    :param matrix2: Second plurality.
    :return: Multiplied matrix.
    """
    result = [[0 for cols in matrix2[0]] for row in matrix1]

    for num1 in range(len(matrix1)):
        for num2 in range(len(matrix2[0])):
            for num_result in range(len(matrix2)):
                result[num1][num2] += matrix1[num1][num_result] * matrix2[num_result][num2]
    return result


def scal_prod(x, y):
    n = len(x)  # определяем длины списков !!!
    s = 0
    for i in range(n):
        s += x[i]*y[i]
    return s


def matrix_single_mul(matrix, n):
    """
    Multiply matrix on simple number.
    :param matrix: First plurality.
    :param n: Second plurality.
    :return: Multiplied matrix.
    """
    result = [[number * n for number in line] for line in matrix]
    return result


def matrix_pow(matrix, n):
    """
    Increment matrix on n.
    :param matrix: Matrix.
    :param n: Increment index.
    :return: Incremented matrix.
    """
    result = [[number ** n for number in line] for line in matrix]
    return result


def matrix_sum(matrix1, matrix2):
    """
    Sum of two matrix.
    :param matrix1: First synopsis.
    :param matrix2: Second synopsis.
    :return: Summarized matrix.
    """
    result = [[matrix1[line][row] + matrix2[line][row] for row in range(len(matrix1[line]))] for line in
              range(len(matrix1))]
    return result


def matrix_dif(matrix1, matrix2):
    """
    Dif of two matrix.
    :param matrix1:
    :param matrix2:
    :return: Result of operation.
    """
    result = [[matrix1[line][row] - matrix2[line][row] for row in range(len(matrix1[line]))] for line in
              range(len(matrix1))]
    return result


def transpose(matrix):
    """
    Transpose matrix.
    :param matrix: Matrix.
    :return: Transposed matrix.
    """
    result = [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]
    return result


def load_pre_trined_neurons(file_name):
    """
    Ask about loading pre-trained neuron.
    :param file_name: Name of file.
    :return: User answer about loading or not pretrined neuro.
    """
    try:
        inpt = str(input(f'Do you want to load pre-trined neurons {file_name}1.npy and {file_name}2.npy? (Y/n)'))
    except Exception as err:
        print(err)
    if inpt == 'n' or inpt == 'N':
        return False
    return True


def get_current_dir():
    """

    """
    current_dir = '/'.join(__file__.split('/')[:-1])
    return current_dir


def get_filename(path):
    """

    """
    filename = path.split('/')[-1].split('.')[0]
    return filename


def is_pre_trained(image_name):
    """

    """
    global pre_trained_files
    current_dir = get_current_dir()
    for dirname, _, filename in os.walk(f'{current_dir}/pre-trained'):
        pre_trained_files = filename

    if f'{image_name}1.npy' in pre_trained_files and f'{image_name}2.npy' in pre_trained_files:
        return True
    return False


def save_compress_image(save_file, compressed_matrix):
    """

    """
    with open(save_file, 'wb') as compr_file:
        np.save(compr_file, compressed_matrix)
