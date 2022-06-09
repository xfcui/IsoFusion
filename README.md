## Peptide feature detection based on multitask fusion learning

```
Authors: Qihong Jiao^, Yuxiao Wang^, Shiwei Sun*, Xuefeng Cui*
    - ^: These authors contributed equally to this work.
    - *: To whom correspondence should be addressed.
Contact: xfcui@email.sdu.edu.cn
Publish: https://xxx.org/xxx
```


### Introduction

We propose a model based on deep learning, called IsoFusion. This method does not rely on expert knowledge and does not need to adjust complex parameters, which makes it easier to use than traditional methods. In addition, our model is an end-to-end model that can predict charge, number of isotopes and retention time directly from the mass spectrum. The main contributions of our work are listed as followings.: (a) A novel end-to-end model for peptide feature detection from mass spectrum were proposed. (b) FuseBlock that integrates features from different dimensions. (c) Using the multi-task learning to predict charge, number of isotopes and retention time simultaneously, the auxiliary task can help improve the learning performance of the main task.

![](IsoFusion/img/Overall.jpg)


### Usage

You will need to convert your raw mass spectrometry files to MS1 format. The conversion tool can use MSConvert, which you need to download and install yourself.

Suppose you have MS1 files now, and then, you can directly use the Docker Image we provide, or you can build your own running environment:


#### Run with docker

Pull the docker image: `docker pull jorhelp/isofusion`, for users in Mainland China: `docker pull registry.cn-hangzhou.aliyuncs.com/sdu-bioinfo/isofusion`

Run:
```shell
docker run --name isofusion --runtime=nvidia  -v PATH_TO_MS1:/mnt isofusion python3 -Bu IsoFusion/run_IsoFusion.py --file /mnt/MS1_FILE --output /mnt/ --process_num 8 --gpu 0 --batch_size 512
```


#### Run without docker

Clone this repository by:
```shell
git clone https://github.com/xfcui/IsoFusion.git
```

**Make sure the python version you use is >= 3.7**, and install the packages by:
```shell
pip install -r requirements.txt
```

Optional arguments:
```shell
-h, --help                show this help message and exit
--file FILE               the target file (absolute path)
--output OUTPUT           the dir where results will be saved
--process_num PROCESS_NUM multiprocess
--gpu GPU                 specify the gpu num you use
--batch_size BATCH_SIZE   batch size will be used
```

Run:
```shell
./run_IsoFusion.py --file ~/dataset/ms1/***.ms1 --process_num 8 --gpu 0 --batch_size 256
```
