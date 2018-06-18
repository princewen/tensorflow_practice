def hit(gt_items, pred_items):
    count = 0
    for item in pred_items:
        if item in gt_items:
            count += 1
    return count
