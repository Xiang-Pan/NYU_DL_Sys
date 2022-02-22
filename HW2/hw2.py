'''
Author: Xiang Pan
Date: 2022-02-22 00:42:01
LastEditTime: 2022-02-22 00:45:49
LastEditors: Xiang Pan
Description: 
FilePath: /NYU_DL_Sys/HW2/hw2.py
@email: xiangpan@nyu.edu
'''
# Xavier initialization (also called Glorot initialization) was developed with aim to solve gradient vanishing problem, but for relu activation function, it is not necessary. Compaering the performance of Xavier initialization and He initialization for ReLU, He initialization is better, becuase the input and output is within the same range and scale.