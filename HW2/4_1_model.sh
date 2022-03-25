###
 # @Author: Xiang Pan
 # @Date: 2022-03-07 01:54:43
 # @LastEditTime: 2022-03-07 01:57:07
 # @LastEditors: Xiang Pan
 # @Description: 
 # @FilePath: /HW2/4_1_model.sh
 # @email: xiangpan@nyu.edu
### 
python 4_1.py --number_layers=2 --config_number 0 &
python 4_1.py --number_layers=2 --config_number 1 &
python 4_1.py --number_layers=2 --config_number 2 &
python 4_1.py --number_layers=2 --config_number 3 &
python 4_1.py --number_layers=2 --config_number 4 &
wait;
python 4_1.py --number_layers=3 --config_number 0 &
python 4_1.py --number_layers=3 --config_number 1 &
python 4_1.py --number_layers=3 --config_number 2 &
python 4_1.py --number_layers=3 --config_number 3 &
wait;
python 4_1.py --number_layers=4 --config_number 0 &
python 4_1.py --number_layers=4 --config_number 1 &
python 4_1.py --number_layers=4 --config_number 2 &
wait;
