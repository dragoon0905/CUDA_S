# CUDA in semantic segmentation

## Problem definition 
![image](https://user-images.githubusercontent.com/33536599/174297258-5ffafc19-f632-4bb2-8fb6-d2d61941c81e.png)

## Main contribution
1. Forward Positive Transfer (FPT)  
 To transfer the knowledge from the previous tasks to the current task, we utilize a self-training with the pseudo-best label for the current task by mining the confident information in each task.


2. Backward Positive Transfer (BPT) 
 During an additional training step, we adopt a rehearsal method to alleviate forgetting and utilize self-training with pseudo-best label for the previous task to further improve the performance of the previous tasks using the current task knowledge.


## Method

### + TM[1] (Target-specific Memory)
![image](https://user-images.githubusercontent.com/33536599/174299080-65ab91ca-2129-4e07-b6d5-1a6284cbca1a.png)

### + Baseline-- Pixmatch[2]
![image](https://user-images.githubusercontent.com/33536599/174299226-1424c901-197b-4995-807d-078034995458.png)

###  + Forward Positive Transfer (FPT) framework
![image](https://user-images.githubusercontent.com/33536599/174299691-be645625-6fad-44bd-a4d0-701fc46b96e9.png)

###  + Backward Positive Transfer (BPT) framework
![image](https://user-images.githubusercontent.com/33536599/174299532-d411d2fd-100d-4a62-8567-08eebd7afe2c.png)

### +전체 학습 framework
![image](https://user-images.githubusercontent.com/33536599/174300287-9d7a3e48-16ad-43d2-9429-d5d22bdd6ce2.png)

## Expreimental setting
![image](https://user-images.githubusercontent.com/33536599/174300403-89855059-df81-4382-a8d1-6303a4c7b86b.png)

## Result Table
![image](https://user-images.githubusercontent.com/33536599/174300547-954073da-3aa6-4bd3-88cd-f67822a92dfc.png)
[1] Tsai, Yi-Hsuan, et al. "Learning to adapt structured output space for semantic segmentation." Proceedings of the IEEE conference on computer vision and pattern recognition. 2018.
[2] Wu, Zuxuan, et al. "Ace: Adapting to changing environments for semantic segmentation." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2019.
[3] Melas-Kyriazi, Luke, and Arjun K. Manrai. "PixMatch: Unsupervised domain adaptation via pixelwise consistency training." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021
![image](https://user-images.githubusercontent.com/33536599/174300575-83576b37-e33c-43e4-87fe-8a928ed2e1ae.png)


