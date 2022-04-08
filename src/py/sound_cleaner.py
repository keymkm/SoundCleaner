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
DEFAULT_WEIGHTS_DIR = os.path.join(ROOT_DIR, 'weights', '2022-04-08-083745_dae_denoiser_best_80x80_loss099')


class SoundCleaner:
    """Класс для подавления шума в mel-спектрограмме"""

    def __init__(self,
                 weights=DEFAULT_WEIGHTS_DIR,
                 speclength=80,
                 specheight=80):
        """
        :param weights: путь к директории с весами модели для шумоподавления в спектрограмме
        :param speclength: максимальная длина спектрограммы
        :param specheight: высота спектрограммы

        """
        self.model = tf.keras.models.load_model(weights)
        self.spec_max_length = speclength
        self.spec_height = specheight

    # init

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

    def transform_spec(self, p_spec):
        """
        Преобразование спектрограммы к виду, необходимому для подачи в модель
        :param p_spec: спектрограмма numpy-массив размерности (<длина>, spec_height)

        :return spec_parts: спектрограмма в формате (<количество частей>, spec_max_length, spec_height, 1)
        :return padding_length: длинна добавочной части, необходимой для приведения к размерности модели
        """

        spec = p_spec.copy()
        if spec.dtype != 'float32':
            spec = p_spec.astype('float32')
        if len(spec.shape) == 2:
            spec = np.expand_dims(spec, axis=2)

        # количество частей, на которые надо разделить спектрограмму для подачи в модель
        parts_cnt = spec.shape[0] // self.spec_max_length
        padding_length = spec.shape[0] % self.spec_max_length
        if padding_length > 0:
            parts_cnt += 1

        spec_parts = np.zeros((parts_cnt, self.spec_max_length, self.spec_height, 1), dtype='float32')
        i = 0
        while i < parts_cnt:
            start = i * self.spec_max_length
            # отрезаем часть до нужного размера
            if (i + 1 == parts_cnt) and (padding_length > 0):
                end = start + padding_length
                sub_spec = spec[start: end, :, :]
                sub_spec = np.append(sub_spec,
                                     np.zeros((self.spec_max_length - sub_spec.shape[0],
                                               spec.shape[1],
                                               spec.shape[2])), axis=0)

            # if

            # добавляем часть до нужного размера
            else:
                end = self.spec_max_length * (i + 1)
                sub_spec = spec[start: end, :, :]
            # else

            spec_parts[i] = sub_spec
            i += 1
        # while

        return spec_parts, padding_length

    # transform_spec

    def restore_spec(self, p_spec, p_padding_length):
        """
        Восстановление спектрограммы к исходному формату, в котором она была передана

        :param p_spec: спектрограмма numpy-массив размерности (spec_parts, spec_max_length, spec_height, 1)
        :param p_padding_length: длинна добавочной части, необходимой для приведения к размерности модели

        :return: спектрограмма numpy-массив размерности (<длина>, spec_height)
        """
        parts_cnt = p_spec.shape[0]
        p_spec = np.reshape(p_spec, (self.spec_max_length * p_spec.shape[0], self.spec_height))
        if p_padding_length > 0:
            p_spec = p_spec[:self.spec_max_length * (parts_cnt - 1) + p_padding_length, :]

        return p_spec

    # restore_spec

    def denoise(self, p_spec_path, p_save_to_file=''):
        """
        Проверка наличия шумов в спектрограмме

        :param p_spec_path: путь к файлу спектрограммы в бинарном формате *.npy,
                       содержащим массив размерности (<длина>, spec_height)
        :param p_save_to_file: путь к файлу, в который сохраняется спектрограмма,
                             если параметр пустой, то спектрограмма не сохраняется в файл

        :return: спектрограмма в формате массива размерности (<длина>, spec_height)
        """
        spec = self.load_spec(p_spec_path)
        spec, padding_length = self.transform_spec(spec)
        spec = self.model.predict(spec)
        spec = self.restore_spec(spec, padding_length)

        if p_save_to_file != '':
            with open(p_save_to_file, 'wb') as f:
                np.save(f, spec)

        return spec

    # denoise

# SoundCleaner

def parse_opt():
    """Разбор параметров командной строки"""

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=DEFAULT_WEIGHTS_DIR, help='weights dir path')
    parser.add_argument('--speclength', type=int, default=80, help='max spectrogram length')
    parser.add_argument('--specheight', type=int, default=80, help='spectrogram height')
    parser.add_argument('--spec', type=str, default='', help='spectrogram file path')
    parser.add_argument('--save', type=str, default='', help='result spectrogram file path')

    return parser.parse_args()

# parse_opt

def run(opt):
    sc = SoundCleaner(weights=opt.weights, speclength=opt.speclength, specheight=opt.specheight)
    sc.denoise(opt.spec, os.path.join(ROOT_DIR, opt.save))
# run

if __name__ == "__main__":
    # !!! Тест
    # noisy_spec = os.path.join(ROOT_DIR, 'data', 'val', 'noisy', '8846', '8846_305209_8846-305209-0019.npy')
    # sc = SoundCleaner()
    # print(sc.denoise(noisy_spec))
    # sc.denoise(noisy_spec, os.path.join(ROOT_DIR, 'data', '8846_305209_8846-305209-0019_clean.npy'))
    # Пример команды python sound_cleaner.py --spec /home/mkm/Projects/GOSZNAK_SoundCleaner/data/val/noisy/8846/8846_305209_8846-305209-0019.npy --save /home/mkm/Projects/GOSZNAK_SoundCleaner/data/8846_305209_8846-305209-0019.npy
    opt = parse_opt()
    run(opt)

