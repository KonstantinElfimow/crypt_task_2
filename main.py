import numpy as np
import random
from matplotlib import pyplot


class Generator:
    def __init__(self, m: int):
        # инициализируем буфер через встроенный rand (для упрощения)
        self.buffer: list = list()
        for _ in range(58):
            self.buffer.append(random.randint(0, 10000))

        # подразумевает под собой n-19
        self.first_index = 39

        # подразумевает под собой n-58
        self.second_index = 0

        # основание (m > 0)
        self.base = m

    # Находим следующий {Xi}
    def next(self) -> int:
        value = (self.buffer[self.second_index] + self.buffer[self.first_index]) % self.base
        del self.buffer[self.second_index]
        self.buffer.append(value)
        return value

    def get_base(self) -> int:
        return self.base


def entropy(labels: list) -> float:
    """ Вычисление энтропии вектора из 0-1 """
    n_labels = len(labels)

    if n_labels <= 1:
        return 0

    counts = np.bincount(labels)
    probs = counts[np.nonzero(counts)] / n_labels
    n_classes = len(probs)

    if n_classes <= 1:
        return 0
    return - np.sum(probs * np.log(probs)) / np.log(n_classes)


def expected_value(X: list, N: int) -> float:
    """мат. ожидание"""
    M: int = 0
    for x in X:
        M += x
    M /= N
    return M


def dispersion(X: list, N: int, M: float) -> float:
    """ Дисперсия (отношение от середины значения) """
    D: int = 0
    for x in X:
        D += (x - M) ** 2
    D /= N
    return D


def show_diagram_plot(X_Y: dict, title: str) -> None:
    pyplot.figure(figsize=(19.20, 10.80))
    pyplot.plot(X_Y.keys(), X_Y.values())
    pyplot.title(title)
    pyplot.show()


def show_bar_plot(X_Y: dict, title: str) -> None:
    pyplot.figure(figsize=(19.20, 10.80))
    pyplot.bar(X_Y.keys(), X_Y.values())
    pyplot.title(title)
    pyplot.show()


def bar_chart(F: list, K: int) -> list:
    """ Гистограмма частотных распределений """
    freq_d: dict = {}
    for f in F:
        # Счётное распределение содержимого массива
        freq_d[int(f * K)] = freq_d.get(int(f * K), 0) + 1
    show_bar_plot(freq_d, f"Гистограмма частотных распределений (полуинтервал [0; 1) разбит на {K} полуинтервалов)")

    freq_l: list = list(freq_d.values())
    return freq_l


def pearson_goodness_of_fit_test(freq: list, N: int, K: int) -> (float, float):
    """ Распределение хи-квадрат. По выборке построим эмпирическое распределение F*(x)
    случайной величины X. Сравнение эмпирического F*(x) и практического F(x) )
    (предполагаемого в гипотезе) производится с помощью специально
    подобранной функции — критерия согласия. """
    x_2: float = 0

    E: float = (N / K)  # ожидаемое число попаданий в j-й интервал

    # распределение хи-квадрат с k-1 степенью свободы
    for f in freq:
        x_2 += ((f - E) ** 2) / E

    return x_2


def random_sequence(N: int) -> list:
    K = 20 if N >= 21 else N // 2  # Количество отрезков для оценки вероятности попадания в полуинтервалы (аj; bj] на
    # [0; 1)
    M = int(input("Введите основание генератора: "))
    generator = Generator(M)

    X: list = list()  # Задаём множество X
    F: list = list()  # значения элементов X на полуинтервале [0; 1) через {fi} = {xi} / m
    for _ in range(N):
        xi = generator.next()
        X.append(xi)
        F.append(xi / M)

    m = expected_value(X, N)
    print(f"Математическое ожидание:\n{m}")

    D = dispersion(X, N, m)
    print(f"Дисперсия:\n{D}")

    freq = bar_chart(F, K)  # Количество наблюдений в j-м интервале элементов множества X
    print(f"Гистограмма частотных распределений:\n{freq}")

    x_2 = pearson_goodness_of_fit_test(freq, N, K)
    print(f"хи-квадрат:\n{x_2}")

    print(f"Случайный ряд:\n{X}")
    print(f"Его энтропия:\n{entropy(X)}")

    return X


if __name__ == '__main__':
    N = int(input("Введите количество членов последовательности: "))
    Q: list = random_sequence(N)
