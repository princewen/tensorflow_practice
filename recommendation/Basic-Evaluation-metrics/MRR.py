import numpy as np


def mrr(gt_items, pred_items):
    for index,item in enumerate(pred_items):
        if item in gt_items:
            return 1/index

