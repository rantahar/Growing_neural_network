import tensorflow as tf
import numpy as np
import time

from dynamic_networks import (
    DynamicModel,
    DynamicDenseLayer,
    DynamicConv2DLayer,
    DynamicConv2DToDenseLayer,
)

#################################################
# A simple test for training the dynamic network
# Builds a standard convolutional model following
# https://www.tensorflow.org/tutorials/images/cnn.
#
# At a constant interval, we update the network
# using the stochastic update step.
#################################################


# General optimization parameters
EPOCHS = 50
IMG_SIZE = 32
batch_size = 100

# Network update parameters
network_updates_every = 10
weight_penalty = 0
cnn_start_features = 4
dense_start_features = 10
new_weight_std = 0.01

BUFFER_SIZE = 100


# Download and process the CIFAR dataset
(train_images, train_labels), (
    valid_images,
    valid_labels,
) = tf.keras.datasets.cifar10.load_data()
n_labels = 10

# Rescale to between 0 and 1
train_images, valid_images = train_images / 255.0, valid_images / 255.0
training_data = (train_images.astype(np.float32), train_labels.astype(np.int32))
valid_data = (valid_images.astype(np.float32), valid_labels.astype(np.int32))

# Build shuffled and batched datasets
train_dataset = (
    tf.data.Dataset.from_tensor_slices(training_data).shuffle(60000).batch(batch_size)
)
valid_dataset = (
    tf.data.Dataset.from_tensor_slices(valid_data).shuffle(10000).batch(batch_size)
)


# Create two dynamic dense layers
layers = [
    DynamicConv2DLayer(3, 3, cnn_start_features, new_weight_std),
    DynamicConv2DLayer(3, cnn_start_features, cnn_start_features, new_weight_std),
    DynamicConv2DLayer(3, cnn_start_features, cnn_start_features, new_weight_std),
    DynamicConv2DLayer(3, cnn_start_features, cnn_start_features, new_weight_std),
    DynamicConv2DToDenseLayer(2 * 2, cnn_start_features, dense_start_features, new_weight_std),
    DynamicDenseLayer(dense_start_features, dense_start_features, new_weight_std),
    DynamicDenseLayer(dense_start_features, 10, new_weight_std),
]
classifier = DynamicModel(layers, new_weight_std=new_weight_std)



# The loss function
# This is the full loss for the gradient descent.
# the network update step includes a further weight
# penalty
def compute_loss(data):
    predictions = classifier(data[0])
    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(data[1][:, 0], predictions)
    )
    return loss


# Update weights using Adam
optimizer = tf.optimizers.Adam()


def gradient_train_step(data):
    trainable_variables = classifier.trainable_variables()

    with tf.GradientTape() as tape:
        loss = compute_loss(data)

    gradients = tape.gradient(loss, trainable_variables)

    classifier.apply_adam(gradients)
    return loss


time_elapsed = 0

valid_iterator = iter(valid_dataset.repeat().shuffle(BUFFER_SIZE))

# The update loop
for epoch in range(1, EPOCHS + 1):
    start_time = time.time()
    network_changes = 0

    # Run training over all batches.
    train_loss = 0
    for i, element in enumerate(train_dataset):
        if (i + 1) % network_updates_every == 0:
            # network update step
            valid_element = valid_iterator.next()
            network_changes += classifier.update_features(
                valid_element, compute_loss, weight_penalty
            )
            classifier.prune(0.01)

        # standard gradient update step
        loss = gradient_train_step(element)
        train_loss += loss.numpy()
    train_loss *= batch_size / train_images.shape[0]
    end_time = time.time()

    # Print the state of the network
    classifier.summary()

    # Calculate validation loss.
    valid_loss = 0
    for element in valid_dataset:
        loss = compute_loss(element)
        valid_loss += loss.numpy()
    valid_loss *= batch_size / valid_images.shape[0]
    print(
        "Epoch {} done in {} seconds, loss {}, validation loss {}, network changes {}".format(
            epoch, end_time - start_time, train_loss, valid_loss, network_changes
        )
    )

    time_elapsed += end_time - start_time
    print("Time elapsed {}".format(time_elapsed))

