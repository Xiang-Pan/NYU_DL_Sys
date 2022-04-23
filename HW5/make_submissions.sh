###
 # @Created by: Xiang Pan
 # @Date: 2022-04-09 04:47:31
 # @LastEditors: Xiang Pan
 # @LastEditTime: 2022-04-22 22:15:21
 # @Email: xiangpan@nyu.edu
 # @FilePath: /HW5/make_submissions.sh
 # @Description: 
### 
echo "Start to make submissions"
hw_name=basename $PWD

# get all the problems
problems=$(ls -d problem*.ipynb)
echo "problems: $problems"
for problem_ipynb in $problems
do
    jupyter-nbconvert $problem_ipynb --to pdf
done
rm -rf ./submissions
mkdir -p ./submissions

cp *.ipynb ./submissions/     # copy all the ipynb files
cp *.pdf ./submissions/       # copy all the pdf files
cp -r problem* ./submissions/ # copy all the problem folders
cp -r ./README.md ./submissions/
zip -r Intro_DL_Sys_${hw_name}_xp2030.zip ./submissions/