import cv2
import numpy as np
import time

def template_matching(template_path, image_path):
    image = cv2.imread(image_path)
    template = cv2.imread(template_path, 0)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(image_gray, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    return max_val  #вероятность совпадения

def keypoint_matching(template_path, image_path):
    image = cv2.imread(image_path)
    template = cv2.imread(template_path, 0)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create()
    keypoints_template, descriptors_template = orb.detectAndCompute(template, None)
    keypoints_image, descriptors_image = orb.detectAndCompute(image_gray, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors_template, descriptors_image)
    matches = sorted(matches, key=lambda x: x.distance)
    return len(matches)  # количество совпадений

# Путь к изображениям
template_path = 'src/2_crop.jpg'
image_path = 'src/2_input.jpg'

# Измерение времени для Template Matching
start_time = time.time()
template_result = template_matching(template_path, image_path)
template_time = time.time() - start_time
print(f'Template Matching: {template_time:.4f} seconds; Best Match Probability: {template_result:.4f}')

# Измерение времени для Keypoint Matching
start_time = time.time()
keypoint_result = keypoint_matching(template_path, image_path)
keypoint_time = time.time() - start_time
print(f'Keypoint Matching: {keypoint_time:.4f} seconds; Number of Matches: {keypoint_result}')