###
 # @Created by: Xiang Pan
 # @Date: 2022-04-06 03:47:46
 # @LastEditors: Xiang Pan
 # @LastEditTime: 2022-04-06 04:01:50
 # @Email: xiangpan@nyu.edu
 # @FilePath: /HW3/make_submissions.sh
 # @Description: 
### 
find . -name ".DS_Store" -delete
find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf
cp problem*.* submissions/
cp -r problem* submissions/
cp -r models submissions/
zip -r Intro_DL_Sys_HW3_xp2030.zip submissions/