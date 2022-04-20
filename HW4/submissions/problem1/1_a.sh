###
 # @Created by: Xiang Pan
 # @Date: 2022-04-06 20:33:30
 # @LastEditors: Xiang Pan
 # @LastEditTime: 2022-04-06 20:33:31
 # @Email: xiangpan@nyu.edu
 # @FilePath: /HW4/problem1/1_a.sh
 # @Description: 
### 
FILE="$(basename -- $0)"
NAME=`echo "$FILE" | cut -d'.' -f1`
EXTENSION=`echo "$FILE" | cut -d'.' -f2`


python problem1/1_a.py --gpus 0 --lr 0.001 --group $NAME &
