# система, работа с файлами
import os
from pathlib import Path
import argparse

# ML
import tensorflow as tf

# вычисления
import numpy as np

# Константы
# Определение корневой директории проекта
FILE = Path().resolve()
ROOT_DIR = os.path.join(FILE.parents[1])
# Веса по умолчанию
DEFAULT_WEIGHTS_DIR = os.path.join(ROOT_DIR, 'weights', '2022-04-05-133118_cnn_noise_catcher_best_800x80_acc91')


class NoiseCatcher:
    """Класс для определения зашумленности в mel-спектрограмме"""

    def __init__(self, weights=DEFAULT_WEIGHTS_DIR, speclength=800, specheight=80):
        """

        :param weights: путь к директории с весами модели
        :param speclength: максимальная длина спектрограммы

        """
        self.model = tf.keras.models.load_model(weights)
        self.spec_max_length = speclength
        self.spec_height = specheight

    # init

    def transform_spec(self, p_spec):
        """
        Преобразование спектрограммы к виду, необходимому для подачи в модель
        :param p_spec: спектрограмма numpy-массив размерности (<длина>, spec_height)

        :return: спектрограмма в формате (1, spec_max_length, spec_height, 1)
        """

        spec = p_spec.copy()
        if spec.dtype != 'float32':
            spec = p_spec.astype('float32')
        if len(spec.shape) == 2:
            spec = np.expand_dims(spec, axis=2)

        # добавляем до spec_max_length
        if spec.shape[0] < self.spec_max_length:
            spec = np.append(spec,
                             np.zeros((self.spec_max_length - spec.shape[0],
                                       spec.shape[1],
                                       spec.shape[2])), axis=0)

        # обрезаем до spec_max_length
        elif spec.shape[0] > self.spec_max_length:
            spec = spec[:self.spec_max_length, :, :]

        spec = np.expand_dims(spec, axis=0)

        return spec

    # transform_spec

    @staticmethod
    def load_spec(p_spec_path):
        """
        загрузка спетрограммы из файла

        :param p_spec_path: путь к файлу спектрограммы в бинарном формате *.npy
        :return: спектрограмма
        """
        with open(p_spec_path, 'rb') as f:
            spec = np.load(f).astype('float32')

        return spec

    # load_spec

    def is_noisy(self, p_spec_path):
        """
        Проверка наличия шумов в спектрограмме

        :param p_spec_path: путь к файлу спектрограммы в бинарном формате *.npy,
                       содержащим массив размерности (<длина>, spec_height)
        :return: True - в спектрограмме обнаружены шумы, иначе False
        """

        proba = self.model.predict(self.transform_spec(self.load_spec(p_spec_path)))

        return bool(round(proba[0][0]))

    # is_noisy

# NoseCatcher


def parse_opt():
    """Разбор параметров командной строки"""

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=DEFAULT_WEIGHTS_DIR, help='weights dir path')
    parser.add_argument('--speclength', type=int, default=800, help='max spectrogram length')
    parser.add_argument('--specheight', type=int, default=80, help='spectrogram height')
    parser.add_argument('--spec', type=str, default='', help='spectrogram file path')

    return parser.parse_args()

# parse_opt


def run(opt):
    nc = NoiseCatcher(weights=opt.weights, speclength=opt.speclength, specheight=opt.specheight)
    return nc.is_noisy(opt.spec)
# run


if __name__ == "__main__":
    # тест
    # nc = NoiseCatcher()
    # clean
    # print(nc.is_noisy(os.path.join(ROOT_DIR, 'data', 'val', 'clean', '8846', '8846_305209_8846-305209-0019.npy')))
    # noisy
    # print(nc.is_noisy(os.path.join(ROOT_DIR, 'data', 'val', 'noisy', '8846', '8846_305209_8846-305209-0019.npy')))

    # Пример команды python noise_catcher.py --spec /home/mkm/Projects/GOSZNAK_SoundCleaner/data/val/clean/720/720_173578_720-173578-0012.npy
    opt = parse_opt()
    run(opt)
