#!/usr/bin/env python3
# coding: utf-8
import numpy as np
import cv2
from trdg.generators import GeneratorFromStrings


def __pil2cv(image):
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    return new_image


def __create_generator(str_list: list):
    generator = GeneratorFromStrings(
        str_list,
        count=len(str_list),
        language="ja",
        size=32,
        skewing_angle=0,
        random_skew=False,
        blur=0,
        random_blur=False,
        background_type=1,
        distorsion_type=0,
        distorsion_orientation=0,
        is_handwritten=False,
        width=-1,
        alignment=1,
        text_color="#000000",
        orientation=0,
        space_width=1.0,
        character_spacing=0,
        margins=(5, 5, 5, 5),
        fit=False,
        output_mask=False,
        word_split=False,
    )
    return generator


def generate_train(target_dict, suffix_dir="xxx"):
    data = list()
    with open(target_dict, mode="r", encoding="utf-8") as f:
        data.extend(f.readlines())
        data = [x.strip() for x in data]

    base_dir = "train_data/" + suffix_dir
    img_dir = f"{base_dir}/train"
    with open(f"{base_dir}/train.txt", mode='w', encoding="utf-8") as f:

        for idx, (img, gt_string) in enumerate(__create_generator(data)):
            img = __pil2cv(img)
            file_path = f"{img_dir}/word_{idx:08}.png"
            cv2.imwrite(file_path, img)
            gt_txt = f"{file_path}\t{gt_string}\n"
            f.write(gt_txt)

            cv2.imshow("img", img)
            print(idx)
            if cv2.waitKey(1) == ord("q"):
                break
        cv2.destroyAllWindows()


def main():
    generate_train(target_dict="nri_train_word_list.txt", suffix_dir="nri")
    # generate_train(target_dict="configs/word_dict/nri.txt", suffix_dir="nri_only")


if __name__ == "__main__":
    main()
