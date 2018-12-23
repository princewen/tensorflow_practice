"""Loss functions."""

import tensorflow as tf


def huber_loss(y_true, y_pred, max_grad=1.):
    """Calculate the huber loss.
    See https://en.wikipedia.org/wiki/Huber_loss
    Parameters
    ----------
    y_true: np.array, tf.Tensor
      Target value.
    y_pred: np.array, tf.Tensor
      Predicted value.
    max_grad: float, optional
      Positive floating point value. Represents the maximum possible
      gradient magnitude.
    Returns
    -------
    tf.Tensor
      The huber loss.
    """
    a = tf.abs(y_true - y_pred)
    less_than_max = 0.5 * tf.square(a)
    greater_than_max = max_grad * (a - 0.5 * max_grad)
    return tf.where(a <= max_grad, x=less_than_max, y=greater_than_max)



def mean_huber_loss(y_true, y_pred, max_grad=1.):
    """Return mean huber loss.
    Same as huber_loss, but takes the mean over all values in the
    output tensor.
    Parameters
    ----------
    y_true: np.array, tf.Tensor
      Target value.
    y_pred: np.array, tf.Tensor
      Predicted value.
    max_grad: float, optional
      Positive floating point value. Represents the maximum possible
      gradient magnitude.
    Returns
    -------
    tf.Tensor
      The mean huber loss.
    """
    return tf.reduce_mean(huber_loss(y_true, y_pred, max_grad=max_grad))


def weighted_huber_loss(y_true, y_pred, weights, max_grad=1.):
    """Return mean huber loss.
    Same as huber_loss, but takes the mean over all values in the
    output tensor.
    Parameters
    ----------
    y_true: np.array, tf.Tensor
      Target value.
    y_pred: np.array, tf.Tensor
      Predicted value.
    weights: np.array, tf.Tensor
      weights value.
    max_grad: float, optional
      Positive floating point value. Represents the maximum possible
      gradient magnitude.
    Returns
    -------
    tf.Tensor
      The mean huber loss.
    """
    return tf.reduce_mean(weights*huber_loss(y_true, y_pred, max_grad=max_grad))