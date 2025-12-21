我有如下目录结构:
imagesTr
imagesVal
labelsTr
labelsVal
scribblesTr

其中各个文件夹下为.nii.gz文件，同个集合（例如训练集）的名称相同。
给我写一个py脚本，把imagesTr, labelsTr, scribblesTr每对应的3个.nii.gz文件存储为一个h5文件，keys为image，label和scribble
同上，imagesVal和labelsVal也存为h5文件，keys为image和label，