import numpy as np



def Kappa_cohen(predictions,groundtruth):
    TP   = 0
    TN   = 0
    FP   = 0
    FN   = 0
    gt_r = 0
    gt_p = 0
    for j,an in enumerate(predictions):
        if(an == groundtruth[j] and an == 1 ):
            TP = TP + 1
        elif(an == groundtruth[j] and an == 0):
            TN = TN + 1
        elif(an != groundtruth[j] and an == 0):
            FN = FN + 1
        elif(an != groundtruth[j] and an == 1):
            FP = FP + 1
        if(groundtruth[j]== 0):
            gt_p = gt_p + 1
        else:
            gt_r = gt_r + 1

    observed_accuracy  =   (TP+TN)/groundtruth.shape[0]
    expected_accuracy  =   ((gt_r*TP)/groundtruth.shape[0] + (gt_p*TN)/groundtruth.shape[0])/groundtruth.shape[0]

    return (observed_accuracy - expected_accuracy)/ (1- expected_accuracy)


def IoU(predictions,groundtruth):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for j,an in enumerate(predictions):
        if(an == groundtruth[j] and an == 1 ):
            TP = TP + 1
        elif(an == groundtruth[j] and an == 0):
            TN = TN + 1
        elif(an != groundtruth[j] and an == 0):
            FN = FN + 1
        elif(an != groundtruth[j] and an == 1):
            FP = FP + 1

    return  TP/(TP+FP+FN)


def Accuracy(predictions,groundtruth):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for j,an in enumerate(predictions):
        if(an == groundtruth[j] and an == 1 ):
            TP = TP + 1
        elif(an == groundtruth[j] and an == 0):
            TN = TN + 1
        elif(an != groundtruth[j] and an == 0):
            FN = FN + 1
        elif(an != groundtruth[j] and an == 1):
            FP = FP + 1

    return  (TP+TN)/groundtruth.shape[0]
