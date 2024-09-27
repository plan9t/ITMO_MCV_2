import cv2
import numpy as np

def keypoint_matching(template_path, image_path):
    # Загрузка изображения и шаблона
    image = cv2.imread(image_path)
    template = cv2.imread(template_path, 0)

    # Преобразование изображения в оттенки серого
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Инициализация ORB детектора
    orb = cv2.ORB_create()

    # Нахождение ключевых точек и дескрипторов
    keypoints_template, descriptors_template = orb.detectAndCompute(template, None)
    keypoints_image, descriptors_image = orb.detectAndCompute(image_gray, None)

    # Создание BFMatcher для сопоставления дескрипторов
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Сопоставление дескрипторов
    matches = bf.match(descriptors_template, descriptors_image)

    # Сортировка совпадений по расстоянию
    matches = sorted(matches, key=lambda x: x.distance)

    # Рисование совпадений на изображении
    result_image = cv2.drawMatches(template, keypoints_template, image, keypoints_image, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Отображение результата
    cv2.imshow('Result', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Пример использования
keypoint_matching('src/8_crop.jpg', 'src/8_similar.jpeg')