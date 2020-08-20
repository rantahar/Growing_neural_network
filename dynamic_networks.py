#################################################
# Implements a dynamical dense layer that allows
# both adding and removing both input and output
# neurons and a simple update step for both.
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

  ### Add a random output neuron
  def expand_out(self):
    old_w = self.w
    old_b = self.b
    new_row =  tf.random.normal((self.input_size, 1), stddev=self.new_weight_std)
    new_bias = tf.random.normal((1,), stddev=self.new_weight_std)
    self.w = tf.Variable(tf.concat([old_w, new_row], 1))
    self.b = tf.Variable(tf.concat([old_b, new_bias], 0))
    self.output_size = self.output_size + 1

  ### Remove a random output neuron
  def contract_out(self, n):
    if self.output_size > 1:
      self.w = tf.Variable(tf.concat([self.w[:,:n], self.w[:,n+1:]], 1))
      self.b = tf.Variable(tf.concat([self.b[:n], self.b[n+1:]], 0))
      self.output_size = self.output_size - 1

  ### Add a random input neuron
  def contract_in(self, n):
    if self.input_size > 1:
      self.w = tf.Variable(tf.concat([self.w[:n], self.w[n+1:]], 0))
      self.input_size = self.input_size - 1

  ### Remove a random input neuron
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



### Add and/or remove neurons between two dynamic layers
# Either attempts to add or remove a neuron.
#
# Add: When a neuron is added, the weights are drawn
#      randomly. The new neuron is kept if it reduces the
#      loss on the current data batch
#
# Remove: A random neuron is chosen. It is removed if this
#         reduces the loss on the current data batch
#
def network_update_step(data_batch, loss_function, nets, weight_penalty = 1e-6):
  
  # Get the current loss, including the weight penalty
  neuron_count = nets[0].output_size
  loss1 = loss_function(data_batch) + weight_penalty*neuron_count*neuron_count
  
  # Make note of the current state
  state1 = nets[0].get_state()
  state2 = nets[1].get_state()

  # Randomly choose wether to add or remove
  if np.random.rand() > 0.5:
    # Adding:
    # expand the number of inputs in first net
    # and the number of outputs in the second
    nets[0].expand_out()
    nets[1].expand_in()

    # Calculate the loss for the new network
    neuron_count = nets[0].output_size
    loss2 = loss_function(data_batch) + weight_penalty*neuron_count*neuron_count

  else:
    # Remove:
    # Choose a random neuron
    n = (int)(nets[0].output_size*np.random.rand())
    # remove it from both networks
    nets[0].contract_out(n)
    nets[1].contract_in(n)

  # Calculate the loss in the new network
  loss2 = loss_function(data_batch) + weight_penalty*neuron_count*neuron_count
  # and the change in the loss
  dloss = loss2-loss1

  # If the loss increases, return to the original state
  if dloss > 0 :
    nets[0].set_state(state1)
    nets[1].set_state(state2)
    accepted = False
  else:
    accepted = True
  
  return accepted



