import tensorflow as tf
import sys

# Launch the graph in a session
sess = tf.Session()
# Runs operations, evaluates tensors in fetches, initialize all variables
sess.run(tf.global_variables_initializer())
# Enable a verbal log inside the shell during the training
# tf.logging.set_verbosity(tf.logging.INFO)


def parser(record):
    keys_to_features = {
        "image_raw": tf.FixedLenFeature([], tf.string),
        "label":     tf.FixedLenFeature([], tf.int64)
    }
    parsed = tf.parse_single_example(record, keys_to_features)
    image = tf.decode_raw(parsed["image_raw"], tf.uint8)
    image = tf.cast(image, tf.float32)
    label = tf.cast(parsed["label"], tf.int32)
    return {"image": image}, label


def input_fn(filenames):
    # Downgrade TF: "num_parallel_reads = 40" doesn't supported anymore in TFRecordDataset's parameters
    dataset = tf.data.TFRecordDataset(filenames=filenames)
    # dataset = tf.contrib.data.TFRecordDataset(filenames=filenames)
    # Downgrade TF: those 2 lines of code aren't supported, replaced by 4 individual fonctions
    # dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(1024, 1))
    # dataset = dataset.apply(tf.contrib.data.map_and_batch(parser, 32))
    dataset = dataset.shuffle(buffer_size=2048)
    dataset = dataset.repeat(1)
    dataset = dataset.map(map_func=parser)
    dataset = dataset.batch(batch_size=32)
    # dataset = dataset.prefetch(buffer_size=1)
    return dataset


def train_input_fn():
    # Return the TFRecords contending the images needed to train the model
    return input_fn(filenames=["train.tfrecords"])


def test_input_fn():
    # Return the TFRecords contending the images needed to test the model
    return input_fn(filenames=["test.tfrecords"])


def val_input_fn():
    # Return the TFRecords contending the images needed to validate the model
    return input_fn(filenames=["val.tfrecords"])


def create_conv_layer(input_node, num_input_channels,
                      num_filters, filter_shape, pool_shape, strides, name):
    # Setup the filter input shape for tf.nn.conv_2d
    conv_filt_shape = [filter_shape[0], filter_shape[1], num_input_channels,
                       num_filters]
    # Initialise weights and bias for the filter
    weights = tf.Variable(tf.truncated_normal(conv_filt_shape, stddev=0.03),
                          name=name+'_W')
    bias = tf.Variable(tf.truncated_normal([num_filters]), name=name+'_b')
    # Setup the convolutional layer operation
    out_layer = tf.nn.conv2d(input_node, weights, [1, 1, 1, 1], padding='SAME')
    # Add the bias
    out_layer += bias
    # Apply a ReLU non-linear activation
    out_layer = tf.nn.relu(out_layer)
    # Set the size of the stride of the sliding window for each dimension
    # of the input tensor
    ksize = [1, pool_shape[0], pool_shape[1], 1]
    # Perform max pooling
    out_layer = tf.nn.max_pool(out_layer, ksize=ksize, strides=strides,
                               padding='SAME')
    return out_layer


def model_fn(features, labels, mode, params):
    # Values used to set the number of output for each layers
    K = 16
    L = 32
    M = 64
    N = 200

    net = features["image"]
    # First layer has to be named "batch_size" to works with Unity3D
    net = tf.reshape(net, [-1, 224, 224, 3])
    net = tf.identity(net, name="input_tensor")
    # 6 convolutionnal layers, perform well with images
    layer1 = create_conv_layer(net, 3, K, [6, 6], [2, 2], [1, 1, 1, 1],
                               name="layer_conv_1")
    layer2 = create_conv_layer(layer1, K, L, [5, 5], [2, 2], [1, 2, 2, 1],
                               name="layer_conv_2")
    layer3 = create_conv_layer(layer2, L, M, [4, 4], [2, 2], [1, 2, 2, 1],
                               name="layer_conv_3")
    layer4 = create_conv_layer(layer3, M, M, [4, 4], [2, 2], [1, 2, 2, 1],
                               name="layer_conv_4")
    layer5 = create_conv_layer(layer4, M, M, [4, 4], [2, 2], [1, 2, 2, 1],
                               name="layer_conv_5")
    layer6 = create_conv_layer(layer5, M, M, [4, 4], [2, 2], [1, 2, 2, 1],
                               name="layer_conv_6")
    # Flatten the last convolutional layer to fit inside a dense layer
    layer_flattened = tf.reshape(layer6, shape=[-1, 7 * 7 * M],
                                 name="layer_reshaped")
    # Dense layer's weights
    W_dense1 = tf.Variable(tf.truncated_normal([7 * 7 * M, N], stddev=0.1),
                           name="W_dense1")
    # Dense layer's bias
    B_dense1 = tf.Variable(tf.ones([N]), name="B_dense1")
    # Dense layer with relu activation
    dense_layer1 = tf.nn.relu(tf.matmul(layer_flattened, W_dense1,
                                        name="layer_flattened") + B_dense1,
                              name="dense_layer1")
    dense_layer1 = tf.layers.dropout(
                   dense_layer1, rate=0.50, noise_shape=None, seed=None,
                   training=(mode == tf.estimator.ModeKeys.TRAIN))
    # Setup the logits' weights and bias to fit the shape of the logits layer
    # N: input, 8 output or number of expected classes
    W_logits = tf.Variable(tf.truncated_normal([N, 8], stddev=0.1),
                           name="W_logits")
    # 8: number of classes
    B_logits = tf.Variable(tf.zeros([8]), name="B_logits")
    # Logit layer: last layer, return the raw values for our predictions
    logits = tf.matmul(dense_layer1, W_logits, name="logits") + B_logits
    predictions = {
                  "classes": tf.argmax(input=logits, axis=1, name = "logits_argmax"),
                  "probabilities": tf.nn.softmax(logits, name="action")
                  }
    # PREDICT: predictions only
    if mode == tf.estimator.ModeKeys.PREDICT:
        spec = tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
        return spec

    # TRAIN & EVAL: Cross-entropy and loss
    c_entro = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
                                                             logits=logits)
    loss = tf.reduce_mean(c_entro)
    # TRAIN: Optimizer with AdamOptimizer
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(params["learning_rate"])
        train_op = optimizer.minimize(loss=loss,
                                      global_step=tf.train.get_global_step())
        spec = tf.estimator.EstimatorSpec(mode=mode, loss=loss,
                                          train_op=train_op)
        return spec
    # EVAL: to mesure the accuracy on the EVAL sample
    eval_metric_ops = {"accuracy": tf.metrics.accuracy(
                                   labels=labels,
                                   predictions=predictions["classes"])}
    spec = tf.estimator.EstimatorSpec(mode=mode, loss=loss,
                                      eval_metric_ops=eval_metric_ops)
    return spec


# Create the estimator, all outputs are writen in "model"
model = tf.estimator.Estimator(model_fn=model_fn,
                               params={"learning_rate": 0.0001},
                               model_dir="./model/")
epoch = 0
while (epoch < 10000):
    # Train the model and set the required number of steps
    model.train(input_fn=train_input_fn, steps=1000)
    # Evaluates the model given evaluation data input_fn and return
    # a dict containing the evaluation metrics specified in model_fn
    result = model.evaluate(input_fn=val_input_fn)
    print(result)
    sys.stdout.flush()
    epoch += 1
