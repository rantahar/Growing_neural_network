#################################################
# Implements a dynamical dense layer that allows
# both adding and removing both input and output
# features and a simple update step for both.
#
# Inspired by "Lifelong Learning with Dynamically
# Expandable Networks", ICLR, 2017 (arXiv:1708.01547)
#################################################

import tensorflow as tf
import numpy as np

# A single dense layer with dynamic input and output size
class dynamic_dense_layer():

  ### Create the layer with a given initial configuration.
  def __init__(self, input_size, output_size, new_weight_std = 0.1):
    if input_size is not None:
      self.w = tf.Variable(tf.random.normal((input_size, output_size), stddev=0.1), trainable=True)
      self.b = tf.Variable(tf.random.normal((output_size,), stddev=0.1), trainable=True)
      self.input_size = input_size
      self.output_size = output_size
      self.new_weight_std = new_weight_std

  ### Initialize from state tuple (or list)
  @classmethod
  def from_state(cls, state, new_weight_std = 0.1):
    obj = cls(None, None)
    obj.w = state[0]
    obj.b = state[1]
    obj.input_size = state[2]
    obj.output_size = state[3]
    obj.new_weight_std = 0.01
    return obj


  ### Add a random output feature
  def expand_out(self):
    new_row =  tf.random.normal((self.input_size, 1), stddev=self.new_weight_std)
    new_bias = tf.random.normal((1,), stddev=self.new_weight_std)
    self.w = tf.Variable(tf.concat([self.w, new_row], 1), trainable=True)
    self.b = tf.Variable(tf.concat([self.b, new_bias], 0), trainable=True)
    self.output_size = self.output_size + 1

  ### Remove a random output feature
  def contract_out(self, n):
    if self.output_size > 1:
      self.w = tf.Variable(tf.concat([self.w[:,:n], self.w[:,n+1:]], 1), trainable=True)
      self.b = tf.Variable(tf.concat([self.b[:n], self.b[n+1:]], 0), trainable=True)
      self.output_size = self.output_size - 1

  ### Add a random input feature
  def contract_in(self, n):
    if self.input_size > 1:
      self.w = tf.Variable(tf.concat([self.w[:n], self.w[n+1:]], 0), trainable=True)
      self.input_size = self.input_size - 1

  ### Remove a random input feature
  def expand_in(self):
    new_column = tf.random.normal((1, self.output_size), stddev=self.new_weight_std)
    self.w = tf.Variable(tf.concat([self.w, new_column], 0), trainable=True)
    self.input_size = self.input_size + 1

  ### Returns a list of trainable variables
  def trainable_variables(self):
    return [self.w, self.b]
  
  ### Returns the current state of the layer
  def get_state(self):
    return (self.w, self.b, self.input_size, self.output_size)
  
  ### Overwrite the current state of the layer with
  # the given state
  def set_state(self, state):
    assert(not isinstance(state[0], tf.Tensor))
    assert(not isinstance(state[1], tf.Tensor))
    self.w = state[0]
    self.b = state[1]
    self.input_size = state[2]
    self.output_size = state[3]

  ### Call the layer
  def __call__(self, inputs):
    assert(self.w.shape == (self.input_size,self.output_size))
    assert(self.b.shape == (self.output_size))
    return tf.matmul(inputs,self.w) + self.b









# A model formed of a number of dynamical dense layers
class dynamic_model():
  
  ### Create the initial model configuration.
  def __init__(self, input_size, output_size, intermediate_layers=0, intermediate_layer_size=8,
               new_weight_std = 0.1, activation = tf.nn.relu):

    ### Create kernels for the convolutions
    self.conv_w = [ 
      tf.Variable(tf.random.normal((3, 3,  3, 32), stddev=0.1), dtype=tf.float32),
      tf.Variable(tf.random.normal((3, 3, 32, 64), stddev=0.1), dtype=tf.float32),
      tf.Variable(tf.random.normal((3, 3, 64, 64), stddev=0.1), dtype=tf.float32)
    ]

    # The first fully connected layer
    connected_input_size = int(input_size[0]*input_size[1]*64 / (4**len(self.conv_w)))
    self.layers = [dynamic_dense_layer(connected_input_size, intermediate_layer_size, new_weight_std)]

    # Intermediate layers
    for n in range(intermediate_layers):
      self.layers += [dynamic_dense_layer(intermediate_layer_size, intermediate_layer_size, new_weight_std)]
    
    # Output layer
    self.layers += [dynamic_dense_layer(intermediate_layer_size, output_size, new_weight_std)]

    # Variables related to fully connected part
    self.new_weight_std = new_weight_std
    self.connected_input_size = connected_input_size
    self.output_size = output_size
    self.activation = activation

  ### Returns the number of weights currently in the model
  def weight_count(self):
    count = 0
    for l in self.layers:
      count += l.input_size*l.output_size + l.output_size
    return count

  def summary(self):
    for i, l in enumerate(self.layers):
      weights = l.input_size*l.output_size + l.output_size
      print("Layer {}: ({},{}),  number weights {}".format(i, l.input_size, l.output_size, weights))

  ### Add a feature
  def expand(self):
    # Pick a layer
    nl = (int)((len(self.layers)-1)*np.random.rand())
    l1 = self.layers[nl]
    l2 = self.layers[nl+1]
    # Expand the number of outputs in the layer
    # and the number of inputs in the next one
    l1.expand_out()
    l2.expand_in()

  ### Add a layer
  def add_layer(self):
    # Pick a layer
    nl = (int)((len(self.layers)-1)*np.random.rand())
    l1 = self.layers[nl]

    # Build an intermediate layer. Start close to one
    stdiv = self.new_weight_std/(l1.output_size)
    new_w = tf.Variable(tf.eye(l1.output_size)+tf.random.normal((l1.output_size, l1.output_size), stddev=stdiv), trainable=True)
    new_b = tf.Variable(tf.random.normal((l1.output_size,), stddev=stdiv), trainable=True)
    new_layer = dynamic_dense_layer.from_state((new_w, new_b, l1.output_size, l1.output_size))
    self.layers.insert(nl+1, new_layer)

  ### Remove a random feature
  def contract(self):
    # Pick a layer
    nl = (int)((len(self.layers)-1)*np.random.rand())
    l1 = self.layers[nl]
    l2 = self.layers[nl+1]

    # Choose a random feature
    n = (int)(self.layers[0].output_size*np.random.rand())
    # remove it from both the layer and the next one
    l1.contract_out(n)
    l2.contract_in(n)

  ### Remove a layer
  # Achieves this by removing the activation step between
  # two layers, producing a linear operation
  def remove_layer(self):
    if len(self.layers) > 2:
      # Pick a layer
      nl = (int)((len(self.layers)-1)*np.random.rand())
  
      # Just drop the activation between the layer and the next,
      # reducing them to a single linear operation
      l1 = self.layers[nl]
      l2 = self.layers[nl+1]

      # Pull the states of the two layers and construct new variables
      st1 = l1.get_state()
      st2 = l2.get_state()
      new_w = tf.Variable(tf.matmul(st1[0],st2[0]), trainable=True)
      new_b = tf.Variable(tf.matmul(tf.expand_dims(st1[1],0),st2[0])[0,:] + st2[1], trainable=True)

      assert(new_w.shape == (l1.input_size, l2.output_size))

      # Build the new layer
      state = [new_w, new_b, l1.input_size, l2.output_size]
      new_layer = dynamic_dense_layer.from_state(state)

      del self.layers[nl]
      del self.layers[nl]
      self.layers.insert(nl, new_layer)
  
  ### Returns a list of trainable variables
  def trainable_variables(self):
    return self.conv_w + [var for l in self.layers for var in l.trainable_variables()]
  
  ### Returns the current state of the model
  def get_state(self):
    return [l.get_state() for l in self.layers]

  ### Overwrite the current state
  def set_state(self, state):
    self.layers = []
    for layer_state in state:
      self.layers += [dynamic_dense_layer.from_state(layer_state)]

  def assert_consistency(self):
    previous_size = self.connected_input_size
    for l in self.layers:
      assert(l.input_size == previous_size)
      assert(l.input_size == l.w.shape[0])
      assert(l.output_size == l.w.shape[1])
      assert(l.output_size == l.b.shape[0])
      previous_size = l.output_size
    assert(self.output_size == previous_size)


  ### Apply the model
  def __call__(self, inputs):
    x = inputs
    for conv_w in self.conv_w:
      x = tf.nn.conv2d(x, conv_w, 2, "SAME")
      x = self.activation(x)
    x = tf.reshape(x, (x.shape[0], -1))
    for l in self.layers[:-1]:
      x = l(x)
      x = self.activation(x)
    x = self.layers[-1](x)
    return x




### Add and/or remove features between two dynamic layers
# Either attempts to add or remove a feature.
#
# Add: When a feature is added, the weights are drawn
#      randomly. The new feature is kept if it reduces the
#      loss on the current data batch
#
# Remove: A random feature is chosen. It is removed if this
#         reduces the loss on the current data batch
#
def network_update_step(data_batch, loss_function, dense_model, 
                        weight_penalty = 1e-9, layer_change_rate = 0.1
                       ):
  
  # Get the current loss, including the weight penalty
  loss1 = loss_function(data_batch) + weight_penalty*dense_model.weight_count()

  # Make note of the current state
  initial_state = dense_model.get_state()

  # Randomly choose wether to add or remove
  if np.random.rand() > 0.5:
    if np.random.rand() > layer_change_rate:
      dense_model.expand()
    else:
      dense_model.add_layer()
  else:
    if np.random.rand() > layer_change_rate:
      dense_model.contract()
    else:
      dense_model.remove_layer()


  # Calculate the loss in the new network
  loss2 = loss_function(data_batch) + weight_penalty*dense_model.weight_count()
  # and the change in the loss
  dloss = loss2-loss1

  #dense_model.summary()
  #print(dloss.numpy(), loss1.numpy(), loss2.numpy())

  # If the loss increases, return to the original state
  if dloss > 0 :
    dense_model.set_state(initial_state)
    accepted = False
  else:
    accepted = True

  dense_model.assert_consistency()
  
  return accepted



