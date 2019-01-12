import numpy as np
import tensorflow as tf
from tqdm import tqdm


def get_model_scores(sess, test_data, neighborhood, input_user_handle, input_item_handle,
                     input_neighborhood_handle, input_neighborhood_length_handle,
                     dropout_handle, score_op, max_neighbors, return_scores=False):
    """
    test_data = dict([positive, np.array[negatives]])
    """
    out = ''
    scores = []
    progress = tqdm(test_data.items(), total=len(test_data),
                    leave=False, desc=u'Evaluate || ')
    for user, (pos, neg) in progress:
        item_indices = list(neg) + [pos]

        feed = {
            input_user_handle: [user] * (len(neg) + 1),
            input_item_handle: item_indices,
        }

        if neighborhood is not None:
            neighborhoods, neighborhood_length = np.zeros((len(neg) + 1, max_neighbors),
                                                          dtype=np.int32), np.ones(len(neg) + 1, dtype=np.int32)

            for _idx, item in enumerate(item_indices):
                _len = min(len(neighborhood[item]), max_neighbors)
                if _len > 0:
                    neighborhoods[_idx, :_len] = neighborhood[item][:_len]
                    neighborhood_length[_idx] = _len
                else:
                    neighborhoods[_idx, :1] = user
            feed.update({
                input_neighborhood_handle: neighborhoods,
                input_neighborhood_length_handle: neighborhood_length
            })

        score = sess.run(score_op, feed)
        scores.append(score.ravel())
        if return_scores:
            s = ' '.join(["{}:{}".format(n, s) for s, n in zip(score.ravel().tolist(), item_indices)])
            out += "{}\t{}\n".format(user, s)
    if return_scores:
        return scores, out
    return scores


def evaluate_model(sess, test_data, neighborhood, input_user_handle, input_item_handle,
                   input_neighborhood_handle, input_neighborhood_length_handle,
                   dropout_handle, score_op, max_neighbors, EVAL_AT=[1, 5, 10]):
    scores = get_model_scores(sess, test_data, neighborhood, input_user_handle, input_item_handle,
                              input_neighborhood_handle, input_neighborhood_length_handle,
                              dropout_handle, score_op, max_neighbors)
    hrs = []
    ndcgs = []
    s = '\n'
    for k in EVAL_AT:
        hr, ndcg = get_eval(scores, len(scores[0]) - 1, k)
        s += "{:<14} {:<14.6f}{:<14} {:.6f}\n".format('HR@%s' % k, hr, 'NDCG@%s' % k, ndcg)
        hrs.append(hr)
        ndcgs.append(ndcg)
    tf.logging.info(s + '\n')

    return hrs, ndcgs


def get_eval(scores, index, top_n=10):
    """
    if the last element is the correct one, then
    index = len(scores[0])-1
    """
    ndcg = 0.0
    hr = 0.0
    assert len(scores[0]) > index and index >= 0

    for score in scores:
        # Get the top n indices
        arg_index = np.argsort(-score)[:top_n]
        if index in arg_index:
            # Get the position
            ndcg += np.log(2.0) / np.log(arg_index.tolist().index(index) + 2.0)
            # Increment
            hr += 1.0

    return hr / len(scores), ndcg / len(scores)
