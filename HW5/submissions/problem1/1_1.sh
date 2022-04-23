###
# @Created by: Xiang Pan
# @Date: 2022-04-22 22:55:33
 # @LastEditors: Xiang Pan
 # @LastEditTime: 2022-04-22 23:03:30
# @Email: xiangpan@nyu.edu
 # @FilePath: /HW5/problem1/1_1.sh
# @Description:
###
python ./third_party/pytorch-ssd/eval_ssd.py \
    --net=mb1-ssd \
    --dataset=./cached_datasets/VOCdevkit/VOC2007 \
    --dataset_type=voc \
    --trained_model=./cached_models/mobilenet-v1-ssd-mp-0_675.pth \
    --label_file=cached_datasets/VOCdevkit/VOC2007/voc-model-labels.txt

# python eval_ssd.py --net mb1-ssd  --dataset ~/data/VOC0712/test/VOC2007/ --trained_model models/mobilenet-v1-ssd-mp-0_675.pth --label_file models/voc-model-labels.txt
