import tensorflow as tf

def sparse_softmax_cce(y_true, y_pred):
    y_true = y_true[:,:,:,0]
    y_true = tf.cast(y_true, 'int32')
    print(y_true.get_shape())
    return tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
