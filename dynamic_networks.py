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


# The dynamic matrix that allows adding and removing features
class dynamic_matrix():
  def __init__(self, shape, std=0.1):
    if shape is not None:
      self.mat  = tf.Variable(tf.random.normal(shape, stddev=std), trainable=True)
      self.mom  = tf.Variable(np.zeros(shape), trainable=False)
      self.mom2 = tf.Variable(np.zeros(shape), trainable=False)

      self.dim = len(shape)
  
  @classmethod
  def from_state(cls, state):
    obj = cls(None)
    obj.mat  = state[0]
    obj.mom  = state[1]
    obj.mom2 = state[2]
    return obj

  ### Add a random output feature
  def expand_out(self, n, std):
    new_row =  tf.random.normal(self.mat.shape[:-1]+(n,), stddev=std)
    self.mat = tf.Variable(tf.concat([self.mat, new_row], self.dim-1), trainable=True)
    # Set momenta for the new row to zero
    mom_row  =  tf.Variable(np.zeros((self.mom.shape[:-1]+(n,))))
    self.mom  = tf.Variable(tf.concat([self.mom, mom_row], self.dim-1), trainable=False)
    mom2_row =  tf.Variable(np.zeros((self.mom2.shape[:-1]+(n,))))
    self.mom2 = tf.Variable(tf.concat([self.mom2, mom2_row], self.dim-1), trainable=False)

  ### Remove a random output feature
  def contract_out(self, n, index):
    if self.shape[-1] > 1:
      start = [0 for x in self.shape]
      size = list(self.shape)
      size[-1] = n*index
      new_mat  = tf.slice(self.mat,  start, size)
      new_mom  = tf.slice(self.mom,  start, size)
      new_mom2 = tf.slice(self.mom2, start, size)

      start[-1] = n*(index+1)
      size[-1] = self.shape[-1] - n*(index+1)
      new_mat  = tf.concat([new_mat,  tf.slice(self.mat, start, size)], self.dim-1)
      new_mom  = tf.concat([new_mom,  tf.slice(self.mom,  start, size)], self.dim-1)
      new_mom2 = tf.concat([new_mom2, tf.slice(self.mom2, start, size)], self.dim-1)

      self.mat  = tf.Variable(new_mat, trainable=True)
      self.mom  = tf.Variable(new_mom, trainable=False)
      self.mom2 = tf.Variable(new_mom2, trainable=False)

  ### Add a random input feature
  def expand_in(self, n, std):
    new_column =  tf.random.normal(self.mat.shape[:-2] + (n,self.mat.shape[-1]), stddev=std)
    self.mat = tf.Variable(tf.concat([self.mat, new_column], self.dim-2), trainable=True)
    # Set momenta for the new row to zero
    mom_column  =  tf.Variable(np.zeros(self.mom.shape[:-2] + (n,self.mom.shape[-1])))
    self.mom  = tf.Variable(tf.concat([self.mom, mom_column], self.dim-2), trainable=False)
    mom2_column =  tf.Variable(np.zeros(self.mom2.shape[:-2] + (n,self.mom2.shape[-1])))
    self.mom2 = tf.Variable(tf.concat([self.mom2, mom2_column], self.dim-2), trainable=False)

  ### Remove a random input feature
  def contract_in(self, n, index):
    if self.mat.shape[-2] > 1:
      start = [0 for x in self.shape]
      size = list(self.shape)
      size[-2] = n*index
      new_mat  = tf.slice(self.mat,  start, size)
      new_mom  = tf.slice(self.mom,  start, size)
      new_mom2 = tf.slice(self.mom2, start, size)

      start[-2] = n*(index+1)
      size[-2] = self.shape[-2] - n*(index+1)
      new_mat  = tf.concat([new_mat,  tf.slice(self.mat, start, size)], self.dim-2)
      new_mom  = tf.concat([new_mom,  tf.slice(self.mom,  start, size)], self.dim-2)
      new_mom2 = tf.concat([new_mom2, tf.slice(self.mom2, start, size)], self.dim-2)

      self.mat  = tf.Variable(new_mat, trainable=True)
      self.mom  = tf.Variable(new_mom, trainable=False)
      self.mom2 = tf.Variable(new_mom2, trainable=False)

  
  def get_state(self):
    return (self.mat,self.mom,self.mom2)

  def set_state(self, state):
    assert(not isinstance(state[0], tf.Tensor))
    assert(not isinstance(state[1], tf.Tensor))
    assert(not isinstance(state[2], tf.Tensor))
    self.mat  = state[0]
    self.mom  = state[1]
    self.mom2 = state[2]

  @property
  def shape(self):
    return self.mat.get_shape().as_list()






# A single dense layer with dynamic input and output size
class dynamic_dense_layer():

  ### Create the layer with a given initial configuration.
  def __init__(self, input_size, output_size, new_weight_std = 0.1):
    if input_size is not None:
      self.w = dynamic_matrix((input_size, output_size), 0.1)
      self.b = dynamic_matrix((1, output_size), 0.1)
      self.dynamic = True
      self.input_size = input_size
      self.output_size = output_size
      self.new_weight_std = new_weight_std

  ### Initialize from state tuple (or list)
  @classmethod
  def from_state(cls, state, new_weight_std = 0.1):
    obj = cls(None, None)
    obj.w = dynamic_matrix.from_state(state[0])
    obj.b = dynamic_matrix.from_state(state[1])
    obj.input_size = state[2]
    obj.output_size = state[3]
    obj.new_weight_std = 0.01
    return obj


  ### Add a random output feature
  def expand_out(self):
    self.w.expand_out(1, self.new_weight_std)
    self.b.expand_out(1, self.new_weight_std)
    self.output_size = self.output_size + 1

  ### Remove a random output feature
  def contract_out(self, index):
    if self.output_size > 1:
      self.w.contract_out(1, index)
      self.b.contract_out(1, index)
      self.output_size = self.output_size - 1

  ### Add a random input feature
  def expand_in(self):
    self.w.expand_in(1, self.new_weight_std)
    self.input_size = self.input_size + 1

  ### Remove a random input feature
  def contract_in(self, index):
    if self.input_size > 1:
      self.w.contract_in(1, index)
      self.input_size = self.input_size - 1

  ### Returns a list of trainable variables
  @property
  def trainable_variables(self):
    return [self.w.mat, self.b.mat]
  
  ### Returns the current state of the layer
  def get_state(self):
    return (self.w.get_state(), self.b.get_state(), self.input_size, self.output_size)

  ### Overwrite the current state of the layer with
  # the given state
  def set_state(self, state):
    assert(not isinstance(state[0], tf.Tensor))
    assert(not isinstance(state[1], tf.Tensor))
    self.w.set_state(state[0])
    self.b.set_state(state[1])
    self.input_size = state[2]
    self.output_size = state[3]

  ### Return the number of weights in the layer
  def weight_count(self):
    return self.input_size*self.output_size + self.output_size

  def summary_string(self):
    return "({}, {})".format(self.input_size, self.output_size)

  ### Apply the layer
  def __call__(self, inputs):
    assert(self.w.shape == [self.input_size,self.output_size])
    assert(self.b.shape == [1, self.output_size])
    return tf.matmul(inputs,self.w.mat) + self.b.mat



# A convolution layer with dynamic filter size
class dynamic_conv2d_layer():

  ### Create the layer with a given initial configuration.
  def __init__(self, width, input_size, output_size, new_weight_std = 0.1):
    if input_size is not None:
      self.w = dynamic_matrix((width, width, input_size, output_size), 0.1)
      self.dynamic = True
      self.width = width
      self.input_size = input_size
      self.output_size = output_size
      self.new_weight_std = new_weight_std

  ### Initialize from state tuple (or list)
  @classmethod
  def from_state(cls, state, new_weight_std = 0.1):
    obj = cls(None, None)
    obj.w = dynamic_matrix.from_state(state[0])
    obj.width = state[1]
    obj.input_size = state[2]
    obj.output_size = state[3]
    obj.new_weight_std = 0.01
    return obj


  ### Add a random output feature
  def expand_out(self):
    self.w.expand_out(1,self.new_weight_std)
    self.output_size = self.output_size + 1

  ### Remove a random output feature
  def contract_out(self, n):
    if self.output_size > 1:
      self.w.contract_out(1, n)
      self.output_size = self.output_size - 1

  ### Remove a random input feature
  def contract_in(self, n):
    if self.input_size > 1:
      self.w.contract_in(1, n)
      self.input_size = self.input_size - 1

  ### Add a random input feature
  def expand_in(self):
    self.w.expand_in(1, self.new_weight_std)
    self.input_size = self.input_size + 1

  ### Returns a list of trainable variables
  @property
  def trainable_variables(self):
    return [self.w.mat]
  
  ### Returns the current state of the layer
  def get_state(self):
    return (self.w.get_state(), self.width, self.input_size, self.output_size)

  ### Overwrite the current state of the layer with
  # the given state
  def set_state(self, state):
    assert(not isinstance(state[0], tf.Tensor))
    self.w.set_state(state[0])
    self.width = state[1]
    self.input_size = state[2]
    self.output_size = state[3]

  ### Return the number of weights in the layer
  def weight_count(self):
    return self.width*self.width*self.input_size*self.output_size

  def summary_string(self):
    return "({}, {}, {}, {})".format(self.width, self.width, self.input_size, self.output_size)

  ### Apply the layer
  def __call__(self, inputs):
    assert(self.w.shape == [self.width, self.width, self.input_size,self.output_size])
    return tf.nn.conv2d(inputs, self.w.mat, 2, "SAME")




# Flattens the output of a conv2d layer and allows adding neurons correctly in
# between
class dynamic_conv2d_to_dense_layer():

  ### Create the layer with a given initial configuration.
  def __init__(self, pixels, features, output_size, new_weight_std = 0.1):
    if pixels is not None:
      self.w = dynamic_matrix((pixels*features, output_size), 0.1)
      self.b = dynamic_matrix((1, output_size), 0.1)
      self.dynamic = True
      self.pixels = pixels
      self.features = features
      self.output_size = output_size
      self.new_weight_std = new_weight_std

  ### Initialize from state tuple (or list)
  @classmethod
  def from_state(cls, state, new_weight_std = 0.1):
    obj = cls(None, None)
    obj.w = dynamic_matrix.from_state(state[0])
    obj.b = dynamic_matrix.from_state(state[1])
    obj.features = state[2]
    obj.output_size = state[3]
    obj.new_weight_std = new_weight_std
    return obj


  ### Add a random output feature
  def expand_out(self):
    self.w.expand_out(1, self.new_weight_std)
    self.b.expand_out(1, self.new_weight_std)
    self.output_size = self.output_size + 1

  ### Remove a random output feature
  def contract_out(self, n):
    if self.output_size > 1:
      self.w.contract_out(1, n)
      self.b.contract_out(1, n)
      self.output_size = self.output_size - 1

  ### Add a random input feature
  def expand_in(self):
    self.w.expand_in(self.pixels, self.new_weight_std)
    self.features = self.features + 1

  ### Remove a random input feature
  def contract_in(self, n):
    if self.features > 1:
      self.w.contract_in(self.pixels, n)
      self.features = self.features - 1

  ### Returns a list of trainable variables
  @property
  def trainable_variables(self):
    return [self.w.mat, self.b.mat]
  
  ### Returns the current state of the layer
  def get_state(self):
    return (self.w.get_state(), self.b.get_state(), self.pixels, self.features, self.output_size)

  ### Overwrite the current state of the layer with
  # the given state
  def set_state(self, state):
    assert(not isinstance(state[0], tf.Tensor))
    assert(not isinstance(state[1], tf.Tensor))
    self.w.set_state(state[0])
    self.b.set_state(state[1])
    self.pixels = state[2]
    self.features = state[3]
    self.output_size = state[4]

  ### Return the number of weights in the layer
  def weight_count(self):
    return self.pixels*self.features*self.output_size + self.output_size

  def summary_string(self):
    return "({}, {}, {})".format(self.pixels, self.features, self.output_size)

  ### Apply the layer
  def __call__(self, inputs):
    assert(self.w.shape == [self.pixels*self.features,self.output_size])
    assert(self.b.shape == [1, self.output_size])
    # Move pixels to the last columns, so that it is easier to add and remove
    x = tf.transpose(inputs, perm=[0, 3, 1, 2])
    # Now flatten
    x = tf.reshape(x, [x.shape[0], -1])
    x = tf.matmul(x,self.w.mat) + self.b.mat
    return x




# A model formed of a number of dynamical dense layers
class dynamic_model():
  
  ### Create the initial model configuration.
  def __init__(self, layers, new_weight_std = 0.1, activation = tf.nn.relu):

    ### Remember the list of layers
    self.layers = layers
    
    # Variables related to fully connected part
    self.new_weight_std = new_weight_std
    self.input_size = self.layers[0].input_size
    self.output_size = self.layers[-1].output_size
    self.activation = activation

  ### Returns the number of weights currently in the model
  def weight_count(self):
    count = 0
    for l in self.layers:
      if l.dynamic:
        count += l.weight_count()
    return count

  def summary(self):
    for i, l in enumerate(self.layers):
      if l.dynamic:
        print("Layer {}: {},  number of weights {}".format(i, l.summary_string(), l.weight_count()))

  ### Add a feature
  def expand(self):
    # Pick a layer
    nl = (int)((len(self.layers)-1)*np.random.rand())
    l1 = self.layers[nl]
    l2 = self.layers[nl+1]
    if not l1.dynamic or not l2.dynamic:
      return

    # Expand the number of outputs in the layer
    # and the number of inputs in the next one
    l1.expand_out()
    l2.expand_in()

  ### Remove a random feature
  def contract(self):
    # Pick a layer
    nl = (int)((len(self.layers)-1)*np.random.rand())
    l1 = self.layers[nl]
    l2 = self.layers[nl+1]
    if not l1.dynamic or not l2.dynamic:
      return

    # Choose a random feature
    n = (int)(l1.output_size*np.random.rand())

    # remove it from both the layer and the next one
    l1.contract_out(n)
    l2.contract_in(n)

  ### Stochastic update: add or remove a feature if it
  ### decreases the loss function
  def update_features(self, data, loss_function, weight_penalty = 1e-9, layer_change_rate = 0.1):
    # Get the current loss, including the weight penalty
    initial_loss = loss_function(data) + weight_penalty*self.weight_count()
    # Make note of the current state
    initial_state = self.get_state()
    # Randomly choose wether to add or remove
    if np.random.rand() > 0.5:
      self.expand()
    else:
      self.contract()
    # Calculate the loss in the new network
    new_loss = loss_function(data) + weight_penalty*self.weight_count()
    # and the change in the loss
    dloss = new_loss-initial_loss
    # If the loss increases, return to the original state
    if dloss > 0 :
      self.set_state(initial_state)
      accepted = False
    else:
      accepted = True

    self.assert_consistency()
    #self.summary()
    return accepted

  ### Returns a list of trainable variables
  def trainable_variables(self):
    return  [var for l in self.layers for var in l.trainable_variables]
  
  ### Returns the current state of the model
  def get_state(self):
    state = []
    for l in self.layers:
      if l.dynamic:
        state.append(l.get_state())
    return state

  ### Overwrite the current state
  def set_state(self, state):
    i=0
    for l in self.layers:
      if l.dynamic:
        l.set_state(state[i])
        i=i+1

  def assert_consistency(self):
    pass
    #previous_size = self.connected_input_size
    #for l in self.layers:
    #  assert(l.input_size == previous_size)
    #  assert(l.input_size == l.w.shape[0])
    #  assert(l.output_size == l.w.shape[1])
    #  assert(l.output_size == l.b.shape[0])
    #  previous_size = l.output_size
    #assert(self.output_size == previous_size)


  ### Apply the model
  def __call__(self, inputs):
    x = inputs
    for l in self.layers[:-1]:
      x = l(x)
      x = self.activation(x)
    x = self.layers[-1](x)
    return x


  #-------------------------------
  # Add or remove dense layers
  #-------------------------------

  ### Add a dense layer
  # The new layer starts close to an identity operation
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



