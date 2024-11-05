import random
import numpy as np


def mask_feature(x):
    T, B, Z = x.shape

    for b in range(B):
        for t in range(0, T, 4):
            random_position = np.random.permutation(np.arange(1,49))[:4] 
            random_position = np.sort(np.append(random_position,[0,49]))
            #p = np.random.randint(0, 4, 5) #每个区域选择的特征
            for i in range(len(random_position)-1):
                # x[t:t+p[i],b,random_position[i]:random_position[i+1]] = 0
                # x[t+p[i]+1:t+4,b,random_position[i]:random_position[i+1]] = 0
                prob = random.random()
                if prob < 0.3:
                    #不变
                    continue
                elif prob < 0.5:
                    #变一个
                    p = np.random.randint(0, 4, 1)
                    x[t+p[0],b,random_position[i]:random_position[i+1]] = 0
                elif prob < 0.7:
                    #变两个
                    p = np.random.randint(0, 4, 2)
                    x[t+p[0],b,random_position[i]:random_position[i+1]] = 0
                    x[t+p[1],b,random_position[i]:random_position[i+1]] = 0
                elif prob <0.9:
                    #变三个
                    p = np.random.randint(0, 4, 3)
                    x[t+p[0],b,random_position[i]:random_position[i+1]] = 0
                    x[t+p[1],b,random_position[i]:random_position[i+1]] = 0
                    x[t+p[2],b,random_position[i]:random_position[i+1]] = 0
                else:
                    x[t:t+4,b,random_position[i]:random_position[i+1]] = 0
    return x

def mask_img(x): #x:[T*B, 9, 84, 84]
    # for i in range(15, 65 ,20):
    #     for j in range(15, 65, 20):
    #         x[:,:,i:i+15,j:j+15] = 0
    for i in range(15, 70 ,21):
        for j in range(15, 70, 21):
            x[:,i:i+13,j:j+13] = 0
    # for i in range(15, 70 ,22):
    #     for j in range(15, 70, 22):
    #         x[:,i:i+11,j:j+11] = 0
    return x
