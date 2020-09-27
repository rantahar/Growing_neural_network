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
      self.dynamic = True
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
  def expand_in(self):
    new_column = tf.random.normal((1, self.output_size), stddev=self.new_weight_std)
    self.w = tf.Variable(tf.concat([self.w, new_column], 0), trainable=True)
    self.input_size = self.input_size + 1

  ### Remove a random input feature
  def contract_in(self, n):
    if self.input_size > 1:
      self.w = tf.Variable(tf.concat([self.w[:n], self.w[n+1:]], 0), trainable=True)
      self.input_size = self.input_size - 1

  ### Returns a list of trainable variables
  @property
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

  ### Return the number of weights in the layer
  def weight_count(self):
    return self.input_size*self.output_size + self.output_size

  def summary_string(self):
    return "({}, {})".format(self.input_size, self.output_size)

  ### Apply the layer
  def __call__(self, inputs):
    assert(self.w.shape == (self.input_size,self.output_size))
    assert(self.b.shape == (self.output_size))
    return tf.matmul(inputs,self.w) + self.b



# A convolution layer with dynamic filter size
class dynamic_conv2d_layer():

  ### Create the layer with a given initial configuration.
  def __init__(self, width, input_size, output_size, new_weight_std = 0.1):
    if input_size is not None:
      self.w = tf.Variable(tf.random.normal((width, width, input_size, output_size), stddev=0.1), trainable=True)
      self.dynamic = True
      self.width = width
      self.input_size = input_size
      self.output_size = output_size
      self.new_weight_std = new_weight_std

  ### Initialize from state tuple (or list)
  @classmethod
  def from_state(cls, state, new_weight_std = 0.1):
    obj = cls(None, None)
    obj.w = state[0]
    obj.width = state[1]
    obj.input_size = state[2]
    obj.output_size = state[3]
    obj.new_weight_std = 0.01
    return obj


  ### Add a random output feature
  def expand_out(self):
    new_row =  tf.random.normal((self.width, self.width, self.input_size, 1), stddev=self.new_weight_std)
    self.w = tf.Variable(tf.concat([self.w, new_row], 3), trainable=True)
    self.output_size = self.output_size + 1

  ### Remove a random output feature
  def contract_out(self, n):
    if self.output_size > 1:
      self.w = tf.Variable(tf.concat([self.w[:,:,:,:n], self.w[:,:,:,n+1:]], 3), trainable=True)
      self.output_size = self.output_size - 1

  ### Remove a random input feature
  def contract_in(self, n):
    if self.input_size > 1:
      self.w = tf.Variable(tf.concat([self.w[:,:,:n], self.w[:,:,n+1:]], 2), trainable=True)
      self.input_size = self.input_size - 1

  ### Add a random input feature
  def expand_in(self):
    new_column = tf.random.normal((self.width, self.width, 1, self.output_size), stddev=self.new_weight_std)
    self.w = tf.Variable(tf.concat([self.w, new_column], 2), trainable=True)
    self.input_size = self.input_size + 1

  ### Returns a list of trainable variables
  @property
  def trainable_variables(self):
    return [self.w]
  
  ### Returns the current state of the layer
  def get_state(self):
    return (self.w, self.width, self.input_size, self.output_size)

  ### Overwrite the current state of the layer with
  # the given state
  def set_state(self, state):
    assert(not isinstance(state[0], tf.Tensor))
    self.w = state[0]
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
    assert(self.w.shape == (self.width, self.width, self.input_size,self.output_size))
    return tf.nn.conv2d(inputs, self.w, 2, "SAME")




# Flattens the output of a conv2d layer and allows adding neurons correctly in
# between
class dynamic_conv2d_to_dense_layer():

  ### Create the layer with a given initial configuration.
  def __init__(self, dense_layer):
    if input_size is not None:
      self.dynamic_input = True
      self.dynamic_output = False
      
  ### An output feature of the conv2 layer has been removed. Remove corresponding
  ### weights from the input of the following dense layer 
  def contract_in(self, n):
    pass

  ### Add a random input feature
  def expand_in(self):
    pass




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
        weights = l.input_size*l.output_size + l.output_size
        print("Layer {}: {},  number weights {}".format(i, l.summary_string(), weights))

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



