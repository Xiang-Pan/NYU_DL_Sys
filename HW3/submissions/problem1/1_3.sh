###
 # @Author: Xiang Pan
 # @Date: 2022-03-26 17:31:39
 # @LastEditTime: 2022-03-26 17:33:06
 # @LastEditors: Xiang Pan
 # @Description: 
 # @FilePath: /HW3/problem1/1_3.sh
 # @email: xiangpan@nyu.edu
### 
python problem1/1_3.py --use_dropout --optimizer_name Adagrad &
python problem1/1_3.py --use_dropout --optimizer_name RMSprop &
python problem1/1_3.py --use_dropout --optimizer_name RMSprop+Nesterov &
python problem1/1_3.py --use_dropout --optimizer_name Adadelta &
python problem1/1_3.py --use_dropout --optimizer_name Adam &