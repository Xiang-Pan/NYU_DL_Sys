###
 # @Created by: Xiang Pan
 # @Date: 2022-04-06 19:51:07
 # @LastEditors: Xiang Pan
 # @LastEditTime: 2022-04-06 20:10:03
 # @Email: xiangpan@nyu.edu
 # @FilePath: /HW4/problem1/1_b.sh
 # @Description: 
### 
FILE="$(basename -- $0)"
NAME=`echo "$FILE" | cut -d'.' -f1`
EXTENSION=`echo "$FILE" | cut -d'.' -f2`

python problem1/1_a.py --gpus 6 --lr 0.01 --group NAME &
python problem1/1_a.py --gpus 7 --lr 0.1 --group NAME &