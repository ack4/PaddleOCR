#!/usr/bin/env python3
# coding: utf-8
import collections


def create_dict():
    read_file = f"train_data/nri/train.txt"
    write_file = f"configs/word_dict/nri.txt"
    with open(read_file, mode='r', encoding="utf-8") as r, open(write_file, mode="w", encoding="utf-8") as w:
        seq_txt = "".join([x.split("\t", 1)[1].strip("\n") for x in r.readlines()])
        uniq = sorted(collections.Counter(seq_txt).keys())
        uniq = [x + "\n" for x in uniq]
        w.writelines(uniq)
        print(f"uniq_size:{len(uniq)}:{uniq}")


def main():
    create_dict()


if __name__ == "__main__":
    main()
