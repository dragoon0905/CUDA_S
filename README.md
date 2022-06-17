# CUDA in semantic segmentation

## Problem definition 
![image](https://user-images.githubusercontent.com/33536599/174297258-5ffafc19-f632-4bb2-8fb6-d2d61941c81e.png)

## Main contribution
1. To transfer the knowledge from the previous tasks to the current task, we utilize a self-training with the pseudo-best label for the current task by mining the confident information in each task.
![image](https://user-images.githubusercontent.com/33536599/174298488-c9390f02-13c8-4bfe-b7da-7519036dd7ac.png)

2. During an additional training step, we adopt a rehearsal method to alleviate forgetting and utilize self-training with pseudo-best label for the previous task to further improve the performance of the previous tasks using the current task knowledge.
![image](https://user-images.githubusercontent.com/33536599/174298523-75b8ce6c-43af-4028-b12c-9030778dbfe6.png)
