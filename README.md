# Growing_neural_network
A neural network layer that grows dynamically during training 


Implements a dynamical dense layer that allows both adding
and removing both input and output features and a simple 
update step for both.

Inspired by "Lifelong Learning with Dynamically Expandable
Networks", ICLR, 2017 (arXiv:1708.01547)

Essentially trades the effort of choosing the number of
features for each layer to two parameters, the number of
network updates per epoch and the weight penalty.


## The dense layer

The 'dynamic_dense' class implements a simple dense layer
with the ability to add and remove features dynamically.

The added weights are drawn from a normal distribution.
Usually this will not decrease the loss, but occationally
it will, especially at the beginning of training.

Removing a feature is straightforward, the associated
weights are removed from the matrix.

Each update results in replacing the tensorflow variables.
As far as I know, any momentum information the gradient
descent algorithm stores will be lost. This is suboptimal,
but when the gradient descent is run for long enough it
does not matter much.


## The update

The 'network_update_step' function attempts to add or remove
a feature between two given dynamical layers.
In both cases the update is only accepted if it decreases the
loss, evaluated on a single batch of data.

The update is stochastic, similar to the gradient descent
update, in the sense that the update weight is evaluated
stochastically.

We add an additional weight penalty to the loss function.
Without the cap, the number of features would in principle
increase without limit as training continues. The weight
penalty adds a soft cap to the number of features.


