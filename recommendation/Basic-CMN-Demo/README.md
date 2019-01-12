# Collaborative Memory Network for Recommendation Systems

https://arxiv.org/pdf/1804.10862.pdf


* Python 3.6
* TensorFlow 1.8+
* dm-sonnet


## Data Format
The structure of the data in the npz file is as follows:

```
train_data = [[user id, item id], ...]
test_data = {userid: (pos_id, [neg_id1, neg_id2, ...]), ...}
```

