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
class dynamic_dense():

  ### Create the layer with a given initial configuration.
  def __init__(self, input_size, output_size, new_weight_std = 0.1):
    self.w = tf.Variable(tf.random.normal((input_size, output_size), stddev=0.1))
    self.b = tf.Variable(tf.random.normal((output_size,), stddev=0.1))
    self.input_size = input_size
    self.output_size = output_size
    self.new_weight_std = 0.01

  ### Add a random output feature
  def expand_out(self):
    old_w = self.w
    old_b = self.b
    new_row =  tf.random.normal((self.input_size, 1), stddev=self.new_weight_std)
    new_bias = tf.random.normal((1,), stddev=self.new_weight_std)
    self.w = tf.Variable(tf.concat([old_w, new_row], 1))
    self.b = tf.Variable(tf.concat([old_b, new_bias], 0))
    self.output_size = self.output_size + 1

  ### Remove a random output feature
  def contract_out(self, n):
    if self.output_size > 1:
      self.w = tf.Variable(tf.concat([self.w[:,:n], self.w[:,n+1:]], 1))
      self.b = tf.Variable(tf.concat([self.b[:n], self.b[n+1:]], 0))
      self.output_size = self.output_size - 1

  ### Add a random input feature
  def contract_in(self, n):
    if self.input_size > 1:
      self.w = tf.Variable(tf.concat([self.w[:n], self.w[n+1:]], 0))
      self.input_size = self.input_size - 1

  ### Remove a random input feature
  def expand_in(self):
    new_column = tf.random.normal((1, self.output_size), stddev=self.new_weight_std)
    self.w = tf.Variable(tf.concat([self.w, new_column], 0))
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
    self.w = state[0]
    self.b = state[1]
    self.input_size = state[2]
    self.output_size = state[3]

  ### Call the layer
  def __call__(self, inputs):
    return tf.matmul(inputs,self.w) + self.b



# A model formed of a number of dynamical dense layers
class dynamic_dense_model():
  
  ### Create the initial model configuration.
  def __init__(self, input_size, output_size, intermediate_layers=0, new_weight_std = 0.1,
               activation = tf.nn.leaky_relu):
    # Input layer
    self.layers = [dynamic_dense(input_size, 1, new_weight_std)]

    # Intermediate layers
    for n in range(intermediate_layers):
      self.layers += [dynamic_dense(1, 1, new_weight_std)]
    
    # Output layer
    self.layers += [dynamic_dense(1, output_size, new_weight_std)]
    self.activation = activation

  ### Returns the number of weights currently in the model
  def weight_count(self):
    count = 0
    for l in self.layers:
      count += l.input_size*l.output_size + l.output_size
    return count

  ### Add a feature
  def expand(self):
    # Expand the number of inputs in first net
    # and the number of outputs in the second
    self.layers[0].expand_out()
    self.layers[1].expand_in()

  ### Remove a random feature
  def contract(self):
    # Choose a random feature
    n = (int)(self.layers[0].output_size*np.random.rand())
    # remove it from both networks
    self.layers[0].contract_out(n)
    self.layers[1].contract_in(n)
  
  ### Returns a list of trainable variables
  def trainable_variables(self):
    return [var for l in self.layers for var in l.trainable_variables()]
  
  ### Returns the current state of the model
  def get_state(self):
    return [l.get_state() for l in self.layers]

  ### Overwrite the current state
  def set_state(self, state):
    for layer, layer_state in zip(self.layers, state):
      layer.set_state(layer_state)

  ### Apply the model
  def __call__(self, inputs):
    x = inputs
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
def network_update_step(data_batch, loss_function, dense_model, weight_penalty = 1e-9):
  
  # Get the current loss, including the weight penalty
  loss1 = loss_function(data_batch) + weight_penalty*dense_model.weight_count()
  
  # Make note of the current state
  initial_state = dense_model.get_state()

  # Randomly choose wether to add or remove
  if np.random.rand() > 0.5:
    dense_model.expand()
  else:
    dense_model.contract()

  # Calculate the loss in the new network
  loss2 = loss_function(data_batch) + weight_penalty*dense_model.weight_count()
  # and the change in the loss
  dloss = loss2-loss1

  # If the loss increases, return to the original state
  if dloss > 0 :
    dense_model.set_state(initial_state)
    accepted = False
  else:
    accepted = True
  
  return accepted



