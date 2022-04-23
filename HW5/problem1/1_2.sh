###
# @Created by: Xiang Pan
# @Date: 2022-04-22 23:14:54
 # @LastEditors: Xiang Pan
 # @LastEditTime: 2022-04-23 00:41:45
# @Email: xiangpan@nyu.edu
 # @FilePath: /HW5/problem1/1_2.sh
# @Description:
###
python third_party/pytorch-ssd/open_images_downloader.py --root ./cached_datasets/open_images --class_names "Aircraft"


# before finetuning
# python ./third_party/pytorch-ssd/eval_ssd.py \
#     --net=mb1-ssd \
#     --dataset_type=open_images \
#     --dataset=./cached_datasets/open_images \
#     --trained_model=./models/mb1-ssd-Epoch-0-Loss-4.909261781178164.pth \
#     --label_file=models/open-images-model-labels.txt > ./problem1/1_2_before_finetuning.txt

python ./third_party/pytorch-ssd/eval_ssd.py \
    --net=mb1-ssd \
    --dataset=./cached_datasets/VOCdevkit/VOC2007 \
    --dataset_type=voc \
    --trained_model=./cached_models/mobilenet-v1-ssd-mp-0_675.pth \
    --label_file=cached_datasets/VOCdevkit/VOC2007/voc-model-labels.txt > ./problem1/1_2_before_finetuning.txt

# finetuning
python ./third_party/pytorch-ssd/train_ssd.py --dataset_type open_images --datasets ./cached_datasets/open_images --net mb1-ssd --pretrained_ssd ./cached_models/mobilenet-v1-ssd-mp-0_675.pth --scheduler cosine --lr 0.01 --t_max 100 --validation_epochs 5 --num_epochs 100 --base_net_lr 0.001 --batch_size 5


# after finetuning
# python ./third_party/pytorch-ssd/eval_ssd.py \
#     --net=mb1-ssd \
#     --dataset_type=open_images \
#     --dataset=./cached_datasets/open_images \
#     --trained_model=models/mb1-ssd-Epoch-99-Loss-5.095126256514131.pth \
#     --label_file=models/open-images-model-labels.txt > ./problem1/1_2_after_finetuning.txt

python ./third_party/pytorch-ssd/eval_ssd.py \
    --net=mb1-ssd \
    --dataset=./cached_datasets/VOCdevkit/VOC2007 \
    --dataset_type=voc \
    --trained_model=./models/mb1-ssd-Epoch-99-Loss-5.095126256514131.pth \
    --label_file=cached_datasets/VOCdevkit/VOC2007/voc-model-labels.txt > ./problem1/1_2_after_finetuning.txt