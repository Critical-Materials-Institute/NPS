# Reusable subroutines

import tensorflow.compat.v1 as tf

def compatible_run_tf(exprs, **kwargs):
    if not tf.executing_eagerly():
        with tf.Session() as sess:
            return sess.run(exprs, **kwargs)
    else:
        return tuple(map(lambda x: x.numpy(), exprs))

