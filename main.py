import numpy
import matplotlib.pyplot
import matplotlib.image


def save_and_show(array_image):
    read_image = 1 * (array_image + 1) / 2
    figsize = 256 / float(100), 256 / float(100)
    matplotlib.pyplot.figure(figsize=figsize)
    matplotlib.pyplot.axis('off')
    matplotlib.pyplot.imshow(read_image)
    matplotlib.pyplot.show()



def save_weights(first_layer, second_layer):
    arr_reshaped_w1 = first_layer.reshape(first_layer.shape[0], -1)
    arr_reshaped_w2 = second_layer.reshape(second_layer.shape[0], -1)
    numpy.savetxt("save_weights_1.txt", arr_reshaped_w1)
    numpy.savetxt("save_weights_2.txt", arr_reshaped_w2)


def load_weights_():
    loaded_arr_w1, loaded_arr_w2 = numpy.loadtxt("save_weights_1.txt"), numpy.loadtxt("save_weights_2.txt")
    return loaded_arr_w1, loaded_arr_w2


def adaptive_learning_step(matrix):
    tmp = numpy.dot(matrix, numpy.transpose(matrix))
    s = 0
    for i_f in range(len(tmp)):
        s += tmp[i_f]

    return 1.0 / (s * 10)


def img_to_rect(image_h, image_w):
    rectangles = []
    for i in range(image_h // RECT_H):
        for j in range(image_w // RECT_W):
            rect = []
            for y in range(RECT_H):
                for x in range(RECT_W):
                    for color in range(3):
                        rect.append(array_image[i * RECT_H + y, j * RECT_W + x, color])
            rectangles.append(rect)
    return numpy.array(rectangles)


def img_to_rect_for_use(image_h, image_w):
    rectangles = []
    for i in range(image_h // RECT_H_for_use):
        for j in range(image_w // RECT_W_for_use):
            rect = []
            for y in range(RECT_H_for_use):
                for x in range(RECT_W_for_use):
                    for color in range(3):
                        rect.append(array_image_for_use[i * RECT_H_for_use + y, j * RECT_W_for_use + x, color])
            rectangles.append(rect)
    return numpy.array(rectangles)


def training():
    image_h, image_w = numpy.size(array_image, 0), numpy.size(array_image, 1)
    rect_number = int((image_h * image_w) / (RECT_H * RECT_W))
    rectangles = img_to_rect(image_h, image_w).reshape(rect_number, 1, INPUT_LAYER)

    first_layer = numpy.random.rand(INPUT_LAYER, HIDDEN_LAYER) * 2 - 1
    temp = numpy.copy(first_layer)
    second_layer = temp.transpose()

    current_error = MAX_ERROR + 1
    iteration = 0
    while current_error > MAX_ERROR:
        current_error = 0
        iteration += 1
        for rect in rectangles:
            y = rect @ first_layer
            x1 = y @ second_layer
            delta = x1 - rect
            alpha_first = adaptive_learning_step(x1)
            alpha_second = adaptive_learning_step(y)
            first_layer -= alpha_first * numpy.matmul(numpy.matmul(rect.transpose(), delta), second_layer.transpose())
            second_layer -= alpha_second * numpy.matmul(y.transpose(), delta)
        for rect in rectangles:
            y = rect @ first_layer
            x1 = y @ second_layer
            delta = x1 - rect
            current_error += (delta * delta).sum()

        print('Iteration ', iteration, '   ', 'Error ', current_error)

    image_h = numpy.size(array_image, 0)
    image_w = numpy.size(array_image, 1)
    rect_number = int((image_h * image_w) / (RECT_H * RECT_W))

    z = (numpy.size(first_layer, 0) * rect_number) / ((numpy.size(first_layer, 0) + rect_number) * HIDDEN_LAYER + 2),
    print(' z ', z)
    return first_layer, second_layer


def rect_to_matrix(rectangles, image_h, image_w):
    matrix = []
    rect_in_line = image_w // RECT_W
    for i in range(image_h // RECT_H):
        for y in range(RECT_H):
            line = []
            for j in range(rect_in_line):
                for x in range(RECT_W):
                    dot = []
                    for color in range(3):
                        dot.append(rectangles[i * rect_in_line + j, (y * RECT_W * 3) + (x * 3) + color])
                    line.append(dot)
            matrix.append(line)
    return numpy.array(matrix)


def rect_to_matrix_for_use(rectangles, image_h, image_w):
    matrix = []
    rect_in_line = image_w // RECT_W_for_use
    for i in range(image_h // RECT_H_for_use):
        for y in range(RECT_H_for_use):
            line = []
            for j in range(rect_in_line):
                for x in range(RECT_W_for_use):
                    dot = []
                    for color in range(3):
                        dot.append(rectangles[i * rect_in_line + j, (y * RECT_W_for_use * 3) + (x * 3) + color])
                    line.append(dot)
            matrix.append(line)
    return numpy.array(matrix)


def start():
    image_h, image_w = numpy.size(array_image, 0), numpy.size(array_image, 1)
    rect_number = int((image_h * image_w) / (RECT_H * RECT_W))
    rectangles = img_to_rect(image_h, image_w).reshape(rect_number, 1, INPUT_LAYER)
    first_layer, second_layer = training()

    result = []
    for rect in rectangles:
        result.append(rect.dot(first_layer).dot(second_layer))
    result = numpy.array(result)
    save_weights(first_layer, second_layer)
    save_and_show(array_image)
    save_and_show(rect_to_matrix(result.reshape(rect_number, INPUT_LAYER), image_h, image_w))


def start_use():
    image_h, image_w = numpy.size(array_image_for_use, 0), numpy.size(array_image_for_use, 1)
    rect_number = int((image_h * image_w) / (RECT_H_for_use * RECT_W_for_use))
    rectangles = img_to_rect_for_use(image_h, image_w).reshape(rect_number, 1, INPUT_LAYER_for_use)
    first_layer, second_layer = load_weights_()
    result = []
    for rect in rectangles:
        result.append(rect.dot(first_layer).dot(second_layer))
    result = numpy.array(result)
    save_and_show(array_image_for_use)
    save_and_show(rect_to_matrix_for_use(result.reshape(rect_number, INPUT_LAYER_for_use), image_h, image_w))


if __name__ == "__main__":
    main_question = input("1 - Обучение\n"
                          "2 - Использование\n")
    if main_question == "1":
        input_img = input("Введите название картинки: ")
        RECT_H = int(input("Высота блока: "))
        RECT_W = int(input("Ширина блока: "))
        MAX_ALPHA = float(input("Alpha: "))
        HIDDEN_LAYER = int(input("Кол-во нейронов на скрытом слое: "))
        MAX_ERROR = float(input("Максимальная ошибка: "))
        INPUT_LAYER = RECT_H * RECT_H * 3
        array_image = (2.0 * matplotlib.image.imread(input_img) / 1.0) - 1.0
        start()

    if main_question == "2":
        input_img = input("Введите название картинки: ")
        RECT_H_for_use = 8
        RECT_W_for_use = 8
        INPUT_LAYER_for_use = RECT_H_for_use * RECT_H_for_use * 3
        array_image_for_use = (2.0 * matplotlib.image.imread(input_img) / 1.0) - 1.0
        start_use()
