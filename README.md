# Growing Neural Network

An method for updating the hidden layer size of a neural network
dynamically during training. Avoids running a neural architecture
search by allowing the network size to vary.

Inspired by "Lifelong Learning with Dynamically Expandable
Networks", ICLR, 2017 (arXiv:1708.01547)

Essentially trades the effort of choosing the number of
features for each layer to two parameters, the number of
network updates per epoch and the weight penalty.


## The Layers

The ``DynamicDense`` class implements a simple dense layer
with the ability to add and remove features.
The added weights are drawn from a normal distribution.

The ``DynamicConv2DLayer`` class similarly implements a 2D convolutional layer
and the ``DynamicConv2DToDenseLayer`` flattens the output of a convolutional
layer and feeds it to a dense layer.

Each layer implements it's own ADAM update, which is used to run the
gradient descent training step. Note that using Tensorflow implementations
of the gradient descent methods is possible, but momentum information is lost
each time the features are updated.


## The Model

The feature update step is implemented in the ``DynamicModel``.
The model can consist of multiple dynamic and standard Keras layers.
In the feature update step (``Dynamic.update_features()``), one hidden feature
is randomly added or removed between any two connecting dynamic layers.
The updated network is kept if the change reduces the loss on a batch
of training data.

The feature update is stochastic in the sense that the decision to keep the
data is based on a random batch of training data.

Without any additional loss for the number of features in the network, the
size of the network would in principle grow indefinitely, resulting in
overfitting. We add a penalty based on the number of weights, which relates
directly to the computational complexity of the model. This adds a soft cap
on the number of features.

