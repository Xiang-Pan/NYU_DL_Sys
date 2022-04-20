###
 # @Created by: Xiang Pan
 # @Date: 2022-04-09 04:47:31
 # @LastEditors: Xiang Pan
 # @LastEditTime: 2022-04-09 04:48:42
 # @Email: xiangpan@nyu.edu
 # @FilePath: /HW4/make_submissions.sh
 # @Description: 
### 
jupyter-nbconvert ./Problem1.ipynb --to pdf
jupyter-nbconvert ./Problem2.ipynb --to pdf
jupyter-nbconvert ./Problem3.ipynb --to pdf
jupyter-nbconvert ./Problem4.ipynb --to pdf
mkdir -p ./submissions
cp *.ipynb ./submissions/
cp *.pdf ./submissions/
cp -r problem* ./submissions/
cp -r ./README.md ./submissions/
zip -r Intro_DL_Sys_HW4_xp2030.zip ./submissions/