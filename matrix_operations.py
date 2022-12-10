def transpose_matrix(mat):
    convert = list(zip(*mat))
    transpose_matrix = list(map(list, convert))
    return transpose_matrix


def vec_product(vec1: list[int], vec2: list[int]) -> int:
    return sum([int(x * y) for x, y in zip(vec1, vec2)])


def matrix_product(mat1: list[list[int]], mat2: list[list[int]]):
    l, n = len(mat1), len(mat2[0])
    answer = [[0 for i in range(n)] for j in range(l)]
    for i in range(l):
        for j in range(n):
            vec1 = mat1[i]
            vec2 = transpose_matrix(mat2)[j]
            answer[i][j] = vec_product(vec1, vec2)
    return answer
