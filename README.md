# CUDA in semantic segmentation

## Problem definition 
![image](https://user-images.githubusercontent.com/33536599/174297258-5ffafc19-f632-4bb2-8fb6-d2d61941c81e.png)

## Main contribution
1. Forward Positive Transfer (FPT)  
 To transfer the knowledge from the previous tasks to the current task, we utilize a self-training with the pseudo-best label for the current task by mining the confident information in each task.


2.Backward Positive Transfer (BPT) 
 During an additional training step, we adopt a rehearsal method to alleviate forgetting and utilize self-training with pseudo-best label for the previous task to further improve the performance of the previous tasks using the current task knowledge.


## Method

### TM[1] (Target-specific Memory)
![image](https://user-images.githubusercontent.com/33536599/174299080-65ab91ca-2129-4e07-b6d5-1a6284cbca1a.png)

### Baseline-- Pixmatch[2]
![image](https://user-images.githubusercontent.com/33536599/174299226-1424c901-197b-4995-807d-078034995458.png)

###  Forward Positive Transfer (FPT) framework
![image](https://user-images.githubusercontent.com/33536599/174299691-be645625-6fad-44bd-a4d0-701fc46b96e9.png)

###  Backward Positive Transfer (BPT) framework
![image](https://user-images.githubusercontent.com/33536599/174299532-d411d2fd-100d-4a62-8567-08eebd7afe2c.png)



