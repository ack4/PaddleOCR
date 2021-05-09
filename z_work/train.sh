#!/bin/bash -u

# Need to be top dir when execute

# train recognition model
python -m paddle.distributed.launch --gpus 0 tools/train.py -c z_work/configs/rec/jp_lite.yml
python -m paddle.distributed.launch --gpus 0 tools/train.py -c z_work/configs/rec/jp_pretrain.yml
python -m paddle.distributed.launch --gpus 0 tools/train.py -c z_work/configs/rec/jp_nri.yml

# eval recognition model by 1 image
python tools/infer_rec.py -c z_work/configs/rec/jp_lite.yml -o Global.pretrained_model=z_work/checkpoint/jp_lite/latest Global.load_static_weights=false Global.infer_img=z_work/train_data/nri/train/word_00000000.png
python tools/infer_rec.py -c z_work/configs/rec/jp_lite.yml -o Global.load_static_weights=false Global.infer_img=z_work/train_data/nri/train/word_00000000.png

# convert to infer_model
python tools/export_model.py -c z_work/configs/rec/jp_lite.yml -o Global.pretrained_model=z_work/checkpoint/jp_lite/latest Global.load_static_weights=False Global.save_inference_dir=z_work/infer_model/jp_lite
python tools/export_model.py -c z_work/configs/rec/jp_lite.yml -o Global.load_static_weights=False Global.save_inference_dir=z_work/infer_model/jp_lite
python tools/export_model.py -c z_work/configs/rec/jp_pretrain.yml -o Global.load_static_weights=False Global.save_inference_dir=z_work/infer_model/jp_pretrain
python tools/export_model.py -c z_work/configs/rec/jp_nri.yml -o Global.pretrained_model=z_work/checkpoint/jp_nri_only/latest Global.load_static_weights=False Global.save_inference_dir=z_work/infer_model/jp_nri

exit 0