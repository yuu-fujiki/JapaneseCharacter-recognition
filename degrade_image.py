# degrade_image.py

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import cv2
import numpy as np

def degrade_img(name, fnt, pos, offset, rot):
    pil_canvas = Image.new("L", (60, 60), 255)
    pil_img = ImageDraw.Draw(pil_canvas)
    pil_img.text((15 + offset + pos, 15 + offset + pos), chr(int(name)), 0, font=fnt)
    # 画像を回転
    rotated_img = pil_canvas.rotate(rot)
    #OpenCVデータに変換
    ocv_img = np.asarray(rotated_img)
    # Perspectiveゆがみを作る
    trim_positions = np.array([[0, 28], [28, 28], [28, 0], [0, 0]], np.float32)
    pr = np.random.rand(4, 2) * 8 - 4
    pr = pr.astype(np.float32)
    pers_positions = np.array([[15, 43], [43, 43], [43, 15], [15, 15]]).astype(np.float32) + pr
    pers_matrix = cv2.getPerspectiveTransform(pers_positions, trim_positions)
    pers_img = cv2.warpPerspective(ocv_img, pers_matrix, (28, 28))
    #PILデータへ変換
    deg_img = Image.fromarray(pers_img)
    return pers_img
