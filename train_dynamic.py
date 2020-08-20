import tensorflow as tf
import numpy as np
import time

from dynamic_networks import dynamic_dense, network_update_step

#################################################
# A simple test for training the dynamic network
# Builds a standard convolutional model following 
# https://www.tensorflow.org/tutorials/images/cnn.
# Then adds two dynamic dense layers.
#
# The weights are trained using standard methods
# (Adam here). Between each epoch we update the
# network a by randomly adding and/or removing
# neurons. 
#################################################


### General optimization parameters
EPOCHS = 30
IMG_SIZE = 32
batch_size = 100

#### Network update parameters
network_updates_per_epoch = 10
weight_penalty = 1e-6
new_weight_std = 0.1


### Download and process the CIFAR dataset
(train_images, train_labels), (valid_images, valid_labels) = tf.keras.datasets.cifar10.load_data()
n_labels = 10

### Rescale to between 0 and 1
train_images, valid_images = train_images / 255.0, valid_images / 255.0
training_data = (train_images.astype(np.float32), train_labels.astype(np.int32))
valid_data = (valid_images.astype(np.float32), valid_labels.astype(np.int32))

### Build shuffled and batched datasets
train_dataset = tf.data.Dataset.from_tensor_slices(training_data).shuffle(60000).batch(batch_size)
valid_dataset = tf.data.Dataset.from_tensor_slices(valid_data).shuffle(10000).batch(batch_size)



### Create kernels for the convolutions
conv_filter_1 = tf.Variable(tf.random.normal((3, 3,  3, 32), stddev=0.1), dtype=tf.float32)
conv_filter_2 = tf.Variable(tf.random.normal((3, 3, 32, 64), stddev=0.1), dtype=tf.float32)
conv_filter_3 = tf.Variable(tf.random.normal((3, 3, 64, 64), stddev=0.1), dtype=tf.float32)

### Create two dynamic dense layers
ed1 = dynamic_dense(1024, 2, new_weight_std)
ed2 = dynamic_dense(2, 10, new_weight_std)


def classifier(inputs):
  ### Runs the classifier on a batch of images
  x = inputs
  # 32, 32, 3
  x = tf.nn.conv2d(x, conv_filter_1, 2, "SAME")
  x = tf.nn.leaky_relu(x)
  # 16, 16, 32
  x = tf.nn.conv2d(x, conv_filter_2, 2, "SAME")
  x = tf.nn.leaky_relu(x)
  # 8, 8, 64
  x = tf.nn.conv2d(x, conv_filter_3, 2, "SAME")
  x = tf.nn.leaky_relu(x)
  # 4, 4, 64
  x = tf.reshape(x, (x.shape[0], -1))
  # 1024
  x = ed1(x)
  x = tf.nn.leaky_relu(x)
  # N
  x = ed2(x)
  # 10
  return x




### The loss function
# This is the full loss for the gradient descent.
# the network update step includes a further weight
# penalty
def compute_loss(data):
  predictions = classifier(data[0])
  loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(data[1][:,0],predictions))
  return loss




### Update weights using Adam 
optimizer = tf.optimizers.Adam()

def gradient_train_step(data):
  trainable_variables = [conv_filter_1, conv_filter_2, conv_filter_3] + ed1.trainable_variables() + ed2.trainable_variables()

  with tf.GradientTape() as tape:
    tape.watch(trainable_variables)
    loss = compute_loss(data)

  gradients = tape.gradient(loss, trainable_variables)
  optimizer.apply_gradients(zip(gradients, trainable_variables))
  return loss



### The update loop
for epoch in range(1, EPOCHS + 1):
  start_time = time.time()

  # Run a number of network update steps.
  # Each randomly adds or removes a neuron.
  network_changes = 0
  for i, element in enumerate(train_dataset):
    network_changes += network_update_step(element, compute_loss, [ed1, ed2], weight_penalty)
    if i==network_updates_per_epoch:
      break
  print("Neurons {}, number of changes {}".format(ed1.output_size, network_changes))
  
  # Next the standard training step. This runs over all the
  # batches. 
  train_loss = 0
  for element in train_dataset:
    loss = gradient_train_step(element)
    train_loss += loss.numpy()
  train_loss *= batch_size/train_images.shape[0]
  end_time = time.time()

  # Calculate validation loss.
  valid_loss = 0
  for element in valid_dataset:
    loss = compute_loss(element)
    valid_loss += loss.numpy()
  valid_loss *= batch_size/valid_images.shape[0]
  print("Epoch {} done in {} seconds, loss {}, validation loss {}".format(
    epoch, end_time - start_time, train_loss, valid_loss))

