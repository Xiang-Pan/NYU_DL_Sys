###
 # @Created by: Xiang Pan
 # @Date: 2022-04-06 20:01:21
 # @LastEditors: Xiang Pan
 # @LastEditTime: 2022-04-06 20:09:22
 # @Email: xiangpan@nyu.edu
 # @FilePath: /HW4/problem2/2_a.sh
 # @Description: 
### 

# get script name
FILE="$(basename -- $0)"
NAME=`echo "$FILE" | cut -d'.' -f1`
EXTENSION=`echo "$FILE" | cut -d'.' -f2`
python problem1/1_a.py --group $NAME --gpus 3 --lr 0.1 --fix_backbone &
python problem1/1_a.py --group $NAME --gpus 4 --lr 0.001 --fix_backbone &
python problem1/1_a.py --group $NAME --gpus 5 --lr 0.0001 --fix_backbone &
