# CUDA in semantic segmentation

## Problem definition 
![image](https://user-images.githubusercontent.com/33536599/174309282-decefb20-3569-40e8-b202-3bad350840ba.png)

1. 목표는 현재 task에서 좋은 성능을 내고 이전 task에서의 forgetting을 완화하는 것
2. source domain의 성능은 신경쓰지 않으며 target domain의 성능만 고려
3. k 번째 task의 오직 source data와 k 번째 target data만 접근할 수 있다.
4. 모든 domain이 공유하는 class에 대해서만 학습한다.

## Main contribution
1. Forward Positive Transfer (FPT)  
 현재 task를 학습하는데 이전 task의 정보를 활용하기 위해 self-training 할 때 사용하는 pseudo label을 생성 할때 이전 task까지 학습된 reference model을 통과하여 나온 정보를 활용한다. 이때 pseudo label에 사용되는 pixel을 선정은 높은 confidence 값을 기준으로 하였다. 


2. Backward Positive Transfer (BPT) 
 추가적인 학습단계에서 forgetting을 완화하기 위해 memory-based rehearsal 방식을 사용하였다. 이때 현재 task에서 학습된 정보들이 이전 task에 도움을 줄 가능성이 있을 수 있기 때문에 마찬가지로 pseudo-best label을 사용한다. 


## Method

+ ### [TM](https://github.com/joonh-kim/ETM) (Target-specific Memory)
![image](https://user-images.githubusercontent.com/33536599/174299080-65ab91ca-2129-4e07-b6d5-1a6284cbca1a.png)

ETM에서 각 task마다 Domain discrepancy한 정보를 담는 추가적인 subnetwork를 그대로 추가하였습니다.


+ ### Baseline-- [Pixmatch](https://github.com/lukemelas/pixmatch)
![image](https://user-images.githubusercontent.com/33536599/174299226-1424c901-197b-4995-807d-078034995458.png)

Baseline으로 기존 UDA기법인 Pixmatch로 선정하였고 Pixmatch는 source에 대한 segmentation loss와 target에 대한 consistency loss로 구성되어 있습니다.

+ ### Forward Positive Transfer (FPT) framework
![image](https://user-images.githubusercontent.com/33536599/174299691-be645625-6fad-44bd-a4d0-701fc46b96e9.png)

현재 task에서 좋은 성능을 내기 위한 loss 입니다.
현재 task의 이미지를 각 t개의 tm을 통과 시킨 후 threshold 값을 넘고 가장 높은 confidence값을 가지는 pixel들을 추려서 pseudo best를 생성 후 target prediction(TM t)와의 Cross entropy를 이용하는 loss를 추가하였습니다. 


+ ### Backward Positive Transfer (BPT) framework
![image](https://user-images.githubusercontent.com/33536599/174299532-d411d2fd-100d-4a62-8567-08eebd7afe2c.png)

이전 task에서 좋은 성능을 내기 위한 loss 입니다.
샘플링되어 메모리에 저장 되어 있는 이전 task의 이미지를 각 t개의 tm을 통과 시킨 후 threshold 값을 넘고 가장 높은 confidence값을 가지는 pixel들을 추려서 pseudo best를 생성 후 target prediction(TM t)와의 Cross entropy를 이용하는 loss를 추가하였습니다. 

+ ### 전체 학습 framework
![image](https://user-images.githubusercontent.com/33536599/174313447-d924eec1-9ede-4dda-bfd9-048928b8d3cd.png)

training stpe을 2가지로 나눌 수 있다. 처음에는 기존 pixmatch에 FPT loss 만 추가하여 학습을 진행하고 일정 iterration 이후에는 BPT loss를 추가하여 학습했다. 본 코드에서는 30000 iteration 이후부터 BPT loss를 추가하였다.

## Expreimental setting
![image](https://user-images.githubusercontent.com/33536599/174300403-89855059-df81-4382-a8d1-6303a4c7b86b.png)

## Result Table
![image](https://user-images.githubusercontent.com/33536599/174300547-954073da-3aa6-4bd3-88cd-f67822a92dfc.png)

[1] Tsai, Yi-Hsuan, et al. "Learning to adapt structured output space for semantic segmentation." Proceedings of the IEEE conference on computer vision and pattern recognition. 2018.

[2] Wu, Zuxuan, et al. "Ace: Adapting to changing environments for semantic segmentation." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2019.

[3] Melas-Kyriazi, Luke, and Arjun K. Manrai. "PixMatch: Unsupervised domain adaptation via pixelwise consistency training." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021


## Training 
Dependiencies 
+ PyTorch (tested on version 1.9.1, but should work on any version)
+ Hydra 1.1: pip install hydra-core --pre
+ Other: pip install albumentations tqdm tensorboard
+ WandB (optional): pip install wandb

General
학습과 관련된 다양한 parameter들은 config 폴더에 있는 yaml 파일을 통해 설정 후 --config-name=gta5.yaml 호출합니다. 

### Dataset 
+ GTA5 : https://download.visinf.tu-darmstadt.de/data/from_games/
+ SYNTHIA : http://synthia-dataset.net/downloads/ 
+ CityScapes : https://www.cityscapes-dataset.com/
+ IDD : https://idd.insaan.iiit.ac.in/dataset/download/
+ Mapillary Vistas : https://www.mapillary.com/dataset/vistas

### Train
GTA5로 학습된 모델
~~~python
HYDRA_FULL_ERROR=1 python main.py --config-name=gta5 name=gta52city
~~~

### Test
~~~python
HYDRA_FULL_ERROR=1 python main.py --config-name=gta5 train=false name=gta52ciity
~~~

## Acknowledgments
This code is heavily borrowed from [Pixmatch](https://github.com/lukemelas/pixmatch)
