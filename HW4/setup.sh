###
 # @Created by: Xiang Pan
 # @Date: 2022-04-06 18:50:14
 # @LastEditors: Xiang Pan
 # @LastEditTime: 2022-04-06 18:55:56
 # @Email: xiangpan@nyu.edu
 # @FilePath: /HW4/setup.sh
 # @Description: 
### 

if [ ! -d "cached_datasets" ]; then
    mkdir cached_datasets
fi

if [ ! -d "cached_datasets/decathlon-1.0-data.tar" ]; then
    wget http://www.robots.ox.ac.uk/~vgg/share/decathlon-1.0-data.tar.gz -P ./cached_datasets
fi

tar -xzf cached_datasets/decathlon-1.0-data.tar.gz -C ./cached_datasets
tar -xvf ./cached_datasets/aircraft.tar -C ./cached_datasets

