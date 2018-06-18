import numpy as np


label_all = np.random.randint(0,2,[10,1]).tolist()
pred_all = np.random.random((10,1)).tolist()

print(label_all)
print(pred_all)

posNum = len(list(filter(lambda s: s[0] == 1, label_all)))

if (posNum > 0):
    negNum = len(label_all) - posNum
    sortedq = sorted(enumerate(pred_all), key=lambda x: x[1])

    posRankSum = 0
    for j in range(len(pred_all)):
        if (label_all[j][0] == 1):
            posRankSum += list(map(lambda x: x[0], sortedq)).index(j) + 1
    auc = (posRankSum - posNum * (posNum + 1) / 2) / (posNum * negNum)
    print("auc:", auc)