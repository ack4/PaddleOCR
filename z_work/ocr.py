#!/usr/bin/env python3
# coding: utf-8
from PIL import ImageGrab, Image, ImageDraw, ImageFont
from paddleocr import PaddleOCR, draw_ocr
import time
import cv2
import numpy as np
import os
import sys
from glob import glob
from functools import lru_cache

# import pytesseract

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.append("../")


class Levenshtein:
    @staticmethod
    @lru_cache(maxsize=4096)
    def distance(s, t):
        if not s:
            return len(t)
        if not t:
            return len(s)
        if s[0] == t[0]:
            return Levenshtein.distance(s[1:], t[1:])
        l1 = Levenshtein.distance(s, t[1:])
        l2 = Levenshtein.distance(s[1:], t)
        l3 = Levenshtein.distance(s[1:], t[1:])
        return 1 + min(l1, l2, l3)

    @staticmethod
    def standard(s, t):
        return Levenshtein.distance(s, t) / max(len(s), len(t))

    @staticmethod
    def similarity(s, t):
        return -Levenshtein.standard(s, t) + 1


def pil2cv(imgPIL):
    # imgCV_RGB = np.array(imgPIL, dtype=np.uint8)
    imgCV_BGR = np.array(imgPIL)[:, :, ::-1]
    return imgCV_BGR


def cv2pil(imgCV):
    imgCV_RGB = imgCV[:, :, ::-1]
    imgPIL = Image.fromarray(imgCV_RGB)
    return imgPIL


def put_text2(img, text, org, color, size):
    x, y = org
    b, g, r = color
    colorRGB = (r, g, b)
    imgPIL = cv2pil(img)
    draw = ImageDraw.Draw(imgPIL)
    # font = ImageFont.truetype('C:/Windows/Fonts/msgothic.ttc', 18)
    font = ImageFont.truetype('C:/Windows/Fonts/UDDigiKyokashoN-B.ttc', size=size)
    w, h = draw.textsize(text, font=font)
    draw.text(xy=(x, y - h), text=text, fill=colorRGB, font=font)
    imgCV = pil2cv(imgPIL)
    return imgCV


def get_image(glob_images_dir):
    for file in glob(glob_images_dir):
        yield cv2.imread(file)


def get_train(train_txt_path="train_data/nri/train.txt"):
    pass


def draw_result(img: np.ndarray, ocr_result):
    box = [line[0] for line in ocr_result]
    boxes = [np.asarray(x).astype(int) for x in box]
    text = [line[1][0] for line in ocr_result]
    score = [line[1][1] for line in ocr_result]

    img_black = np.zeros_like(img, np.uint8)
    img_black = cv2.fillPoly(img_black, boxes, (0, 255, 0))
    img = cv2.addWeighted(img, 0.6, img_black, 0.3, 0.5)

    for t, b, s in zip(text, boxes, score):
        txt = f"{t} @{round(s * 100)}%"
        img = put_text2(img, txt, (b[0, 0], b[0, 1] + 10), (0, 0, 255), size=18)
    return img


class OCR:
    def __init__(self, rec_model_dir=None, use_gpu=True, det=True, cls=False, rec=True, drop_score=0.5):
        self.__det = det
        self.__cls = cls
        self.__rec = rec
        self.__ocr = PaddleOCR(
            use_gpu=use_gpu,
            gpu_mem=500,
            use_angle_cls=cls,
            det_db_thresh=0.3,
            det_db_box_thresh=0.5,
            det_db_unclip_ratio=2,
            rec_algorithm='CRNN',
            rec_model_dir=rec_model_dir,
            # rec_image_shape="3, 32, 320",
            # rec_char_type='japan',
            rec_batch_num=64,
            max_text_length=25,
            use_space_char=True,
            drop_score=drop_score,
            lang='japan',
        )

    def predict(self, img: np.ndarray, enable_postprocess=False):
        result = self.__ocr.ocr(img, det=self.__det, rec=self.__rec, cls=self.__cls)
        if enable_postprocess:
            img = draw_result(img, result)
            return result, img
        return result

    def predict_rec(self, img: np.ndarray):
        result = self.__ocr.ocr(img, det=self.__det, rec=self.__rec, cls=self.__cls)
        return result


def main():
    # detect_rec()
    rec()


def rec():
    ocr = OCR(rec_model_dir=None, det=False, cls=False, rec=True)
    # ocr = OCR(rec_model_dir="infer_model/jp_pretrain", det=False, cls=False, rec=True)
    # ocr = OCR(rec_model_dir="infer_model/jp_nri", det=False, cls=False, rec=True)

    train_path = "train_data/nri/train.txt"
    with open(train_path, mode="r", encoding="utf-8") as f:
        sim_list = list()
        ok_count = 0
        # w = open("train_data/nri/result/result_new.tsv", mode="w", encoding="utf-8")
        count = 0
        for x in [line.strip().split("\t", 1) for line in f.readlines()]:
            img = cv2.imread(x[0])
            g_txt = x[1]
            result = ocr.predict_rec(img)
            infer_txt = result[0][0]
            sim = Levenshtein.similarity(g_txt, infer_txt)
            sim_list.append(sim)
            # print(f"*** gt:{g_txt}, infer:{infer_txt}, Levenshtein:{ls.distance(g_txt, infer_txt)}")
            txt = f"{count}\t{g_txt}\t{infer_txt}\t{sim}"
            # w.write(txt + "\n")
            print(txt)
            cv2.imshow("img", img)
            if cv2.waitKey(1) == ord('q'):
                break
            if sim == 1.0:
                ok_count += 1
            count += 1
            # if count == 100:
            #     break
        # w.close()
        print("mean_sim:", round(np.asarray(sim_list).mean() * 100, 2))
        print("acc:", round((ok_count / count) * 100, 2))
    cv2.destroyAllWindows()
    exit(0)


def detect_rec():
    # for i in get_image("train_data/rec/eval/*.png"):
    #     print(i.shape)
    # exit(0)

    ocr = OCR(det=True, cls=True, rec=True)

    # ocr = OCR(rec_model_dir="infer_model/jp_lite")
    # ocr = OCR(rec_model_dir="infer_model/jp_pretrain")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FPS, 10)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        # _, img = cap.read()
        # for img in get_image("train_data/rec/train/*.png"):
        img = np.asarray(ImageGrab.grab())
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # data = pytesseract.image_to_string(img, lang='jpn')
        # print(data)
        result, img = ocr.predict(img, enable_postprocess=True)

        img = cv2.resize(img, None, None, fx=0.7, fy=0.7)
        cv2.imshow("img", img)
        if cv2.waitKey(1) == ord('q'):
            break
    cv2.destroyAllWindows()
    cap.release()
    exit(0)


if __name__ == "__main__":
    main()
