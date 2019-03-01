import numpy as np

_Dataset = None
_batch_size = None
_num_negatives = None
_num_items = None
_user_input = None
_item_input = None
_labels = None
_index = None
_num_batch = None
_batch_length = None


def shuffle(dataset, batch_choice, num_negatives):  # negative sampling and shuffle the data

    global _Dataset
    global _batch_size
    global _num_negatives
    global _num_items
    global _user_input
    global _item_input
    global _labels
    global _index
    global _num_batch
    global _batch_length
    _Dataset = dataset
    _num_negatives = num_negatives

    if batch_choice == 'user':
        _num_items, _user_input, _item_input, _labels, _batch_length = _get_train_data_user()
        _num_batch = len(_batch_length)
        return _preprocess(_get_train_batch_user)

    else:
        batch_choices = batch_choice.split(":")
        if batch_choices[0] == 'fixed':
            _batch_size = int(batch_choices[1])
            _num_items, _user_input, _item_input, _labels = _get_train_data_fixed()
            iterations = len(_user_input)
            _index = np.arange(iterations)
            _num_batch = iterations / _batch_size
            return _preprocess(_get_train_batch_fixed)
        else:
            print("invalid batch size !")


def batch_gen(batches, i):
    return [(batches[r])[i] for r in range(4)]


def _preprocess(get_train_batch):  # generate the masked batch list
    user_input_list, num_idx_list, item_input_list, labels_list = [], [], [], []

    for i in range(_num_batch):
        ui, ni, ii, l = get_train_batch(i)
        user_input_list.append(ui)
        num_idx_list.append(ni)
        item_input_list.append(ii)
        labels_list.append(l)
    return (user_input_list, num_idx_list, item_input_list, labels_list)


def _get_train_data_user():
    user_input, item_input, labels, batch_length = [], [], [], []
    train = _Dataset.trainMatrix
    trainList = _Dataset.trainList
    num_items = train.shape[1]
    num_users = train.shape[0]
    for u in range(num_users):
        if u == 0:
            batch_length.append((1 + _num_negatives) * len(trainList[u]))
        else:
            batch_length.append((1 + _num_negatives) * len(trainList[u]) + batch_length[u - 1])
        for i in trainList[u]:
            # positive instance
            user_input.append(u)
            item_input.append(i)
            labels.append(1)
            # negative instances
            for t in range(_num_negatives):
                j = np.random.randint(num_items)
                while j in trainList[u]:
                    j = np.random.randint(num_items)
                user_input.append(u)
                item_input.append(j)
                labels.append(0)
    return num_items, user_input, item_input, labels, batch_length


def _get_train_batch_user(i):
    # represent the feature of users via items rated by him/her
    user_list, num_list, item_list, labels_list = [], [], [], []
    trainList = _Dataset.trainList
    if i == 0:
        begin = 0
    else:
        begin = _batch_length[i - 1]
    batch_index = list(range(begin, _batch_length[i]))
    np.random.shuffle(batch_index)
    for idx in batch_index:
        user_idx = _user_input[idx]
        item_idx = _item_input[idx]
        nonzero_row = []
        nonzero_row += trainList[user_idx]
        num_list.append(_remove_item(_num_items, nonzero_row, item_idx))
        user_list.append(nonzero_row)
        item_list.append(item_idx)
        labels_list.append(_labels[idx])
    user_input = np.array(_add_mask(_num_items, user_list, max(num_list)))
    num_idx = np.array(num_list)
    item_input = np.array(item_list)
    labels = np.array(labels_list)
    return (user_input, num_idx, item_input, labels)


def _get_train_data_fixed():
    user_input, item_input, labels = [], [], []
    train = _Dataset.trainMatrix
    num_items = train.shape[1]
    for (u, i) in train.keys():
        # positive instance
        user_items = []
        user_input.append(u)
        item_input.append(i)
        labels.append(1)
        # negative instances
        for t in range(_num_negatives):
            j = np.random.randint(num_items)
            while train.has_key((u, j)):
                j = np.random.randint(num_items)
            user_input.append(u)
            item_input.append(j)
            labels.append(0)
    return num_items, user_input, item_input, labels


def _get_train_batch_fixed(i):
    # represent the feature of users via items rated by him/her
    user_list, num_list, item_list, labels_list = [], [], [], []
    trainList = _Dataset.trainList
    begin = i * _batch_size
    for idx in range(begin, begin + _batch_size):
        user_idx = _user_input[_index[idx]]
        item_idx = _item_input[_index[idx]]
        nonzero_row = []
        nonzero_row += trainList[user_idx]
        num_list.append(_remove_item(_num_items, nonzero_row, item_idx))
        user_list.append(nonzero_row)
        item_list.append(item_idx)
        labels_list.append(_labels[_index[idx]])
    user_input = np.array(_add_mask(_num_items, user_list, max(num_list)))
    num_idx = np.array(num_list)
    item_input = np.array(item_list)
    labels = np.array(labels_list)
    return (user_input, num_idx, item_input, labels)


def _remove_item(feature_mask, users, item):
    flag = 0
    for i in range(len(users)):
        if users[i] == item:
            users[i] = users[-1]
            users[-1] = feature_mask
            flag = 1
            break
    return len(users) - flag


def _add_mask(feature_mask, features, num_max):
    # uniformalize the length of each batch
    for i in range(len(features)):
        features[i] = features[i] + [feature_mask] * (num_max + 1 - len(features[i]))
    return features
