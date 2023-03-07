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


class DynamicMatrix:
    """The dynamic matrix that allows adding and removing features"""

    def __init__(self, shape, std=0.1):
        self.gradient_step = tf.Variable(0.0, trainable=False)
        if shape is not None:
            self.mat = tf.Variable(tf.random.normal(shape, stddev=std), trainable=True)
            self.mom = tf.Variable(np.zeros(shape).astype("float32"), trainable=False)
            self.mom2 = tf.Variable(np.zeros(shape).astype("float32"), trainable=False)

            self.dim = len(shape)

    @classmethod
    def from_state(cls, state):
        obj = cls(None)
        obj.mat = state[0]
        obj.mom = state[1]
        obj.mom2 = state[2]
        return obj

    def expand_out(self, n, std):
        """Add a random output feature"""

        new_row = tf.random.normal(self.mat.shape[:-1] + (n,), stddev=std)
        self.mat = tf.Variable(
            tf.concat([self.mat, new_row], self.dim - 1), trainable=True
        )

        # Set momenta for the new row to zero
        mom_row = tf.Variable(np.zeros((self.mom.shape[:-1] + (n,))).astype("float32"))
        self.mom = tf.Variable(
            tf.concat([self.mom, mom_row], self.dim - 1), trainable=False
        )
        mom2_row = tf.Variable(
            np.zeros((self.mom2.shape[:-1] + (n,))).astype("float32")
        )
        self.mom2 = tf.Variable(
            tf.concat([self.mom2, mom2_row], self.dim - 1), trainable=False
        )

    def contract_out(self, n, index):
        """Remove a random output feature"""

        if self.shape[-1] > 1:
            start = [0 for x in self.shape]
            size = list(self.shape)
            size[-1] = n * index
            new_mat = tf.slice(self.mat, start, size)
            new_mom = tf.slice(self.mom, start, size)
            new_mom2 = tf.slice(self.mom2, start, size)

            start[-1] = n * (index + 1)
            size[-1] = self.shape[-1] - n * (index + 1)
            new_mat = tf.concat(
                [new_mat, tf.slice(self.mat, start, size)], self.dim - 1
            )
            new_mom = tf.concat(
                [new_mom, tf.slice(self.mom, start, size)], self.dim - 1
            )
            new_mom2 = tf.concat(
                [new_mom2, tf.slice(self.mom2, start, size)], self.dim - 1
            )

            self.mat = tf.Variable(new_mat, trainable=True)
            self.mom = tf.Variable(new_mom, trainable=False)
            self.mom2 = tf.Variable(new_mom2, trainable=False)

    def expand_in(self, n, std):
        """Add a random input feature"""

        new_column = tf.random.normal(
            self.mat.shape[:-2] + (n, self.mat.shape[-1]), stddev=std
        )
        self.mat = tf.Variable(
            tf.concat([self.mat, new_column], self.dim - 2), trainable=True
        )
        # Set momenta for the new row to zero
        mom_column = tf.Variable(
            np.zeros(self.mom.shape[:-2] + (n, self.mom.shape[-1])).astype("float32")
        )
        self.mom = tf.Variable(
            tf.concat([self.mom, mom_column], self.dim - 2), trainable=False
        )
        mom2_column = tf.Variable(
            np.zeros(self.mom2.shape[:-2] + (n, self.mom2.shape[-1])).astype("float32")
        )
        self.mom2 = tf.Variable(
            tf.concat([self.mom2, mom2_column], self.dim - 2), trainable=False
        )

    def contract_in(self, n, index):
        """Remove a random input feature"""

        if self.mat.shape[-2] > 1:
            start = [0 for x in self.shape]
            size = list(self.shape)
            size[-2] = n * index
            new_mat = tf.slice(self.mat, start, size)
            new_mom = tf.slice(self.mom, start, size)
            new_mom2 = tf.slice(self.mom2, start, size)

            start[-2] = n * (index + 1)
            size[-2] = self.shape[-2] - n * (index + 1)
            new_mat = tf.concat(
                [new_mat, tf.slice(self.mat, start, size)], self.dim - 2
            )
            new_mom = tf.concat(
                [new_mom, tf.slice(self.mom, start, size)], self.dim - 2
            )
            new_mom2 = tf.concat(
                [new_mom2, tf.slice(self.mom2, start, size)], self.dim - 2
            )

            self.mat = tf.Variable(new_mat, trainable=True)
            self.mom = tf.Variable(new_mom, trainable=False)
            self.mom2 = tf.Variable(new_mom2, trainable=False)

    def colsum(self, index, treshhold=0.001):
        """Find the L1 sum of a given column
        """
        abs = tf.math.abs(self.mat)
        # There must be a simpler way to slice a single column...
        start = [0 for x in self.shape]
        size = list(self.shape)
        size[-1] = 1
        abs = tf.slice(abs, start, size)
        values = tf.math.reduce_sum(abs, keepdims=False)
        return values

    def get_state(self):
        return (self.mat, self.mom, self.mom2)

    def set_state(self, state):
        assert not isinstance(state[0], tf.Tensor)
        assert not isinstance(state[1], tf.Tensor)
        assert not isinstance(state[2], tf.Tensor)
        self.mat = state[0]
        self.mom = state[1]
        self.mom2 = state[2]

    def apply_adam(self, gradient, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """The Adam gradient descent method"""

        t = self.gradient_step.assign_add(1.0)

        mom = self.mom.assign(beta1 * self.mom + (1 - beta1) * gradient)
        mom2 = self.mom2.assign(beta2 * self.mom2 + (1 - beta2) * gradient * gradient)
        mom_hat = mom / (1 - tf.pow(beta1, t))
        mom2_hat = mom2 / (1 - tf.pow(beta2, t))

        self.mat.assign_add(-alpha * mom_hat / (tf.sqrt(mom2_hat) + epsilon))

    @property
    def shape(self):
        return self.mat.get_shape().as_list()


class DynamicDenseLayer:
    """A single dense layer with dynamic input and output size"""

    def __init__(
            self, input_size, output_size,
            new_weight_std=0.1,
            miminum_input_size=None,
            miminum_output_size=None
    ):
        """Create the layer with a given initial configuration"""

        if input_size is not None:
            self.w = DynamicMatrix((input_size, output_size), new_weight_std)
            self.b = DynamicMatrix((1, output_size), new_weight_std)
            self.dynamic = True
            self.input_size = input_size
            self.output_size = output_size
            self.new_weight_std = new_weight_std

            if miminum_input_size is None:
                self.miminum_input_size = input_size
            else:
                self.miminum_input_size = miminum_input_size

            if miminum_output_size is None:
                self.miminum_output_size = output_size
            else:
                self.miminum_output_size = miminum_output_size


    @classmethod
    def from_state(cls, state, new_weight_std=0.01):
        """Initialize from state tuple (or list)"""

        obj = cls(None, None)
        obj.w = DynamicMatrix.from_state(state[0])
        obj.b = DynamicMatrix.from_state(state[1])
        obj.input_size = state[2]
        obj.output_size = state[3]
        obj.new_weight_std = new_weight_std
        return obj

    def expand_out(self):
        """Add a random output feature"""

        self.w.expand_out(1, self.new_weight_std)
        self.b.expand_out(1, self.new_weight_std)
        self.output_size = self.output_size + 1

    def contract_out(self, index):
        """Remove a random output feature"""

        if self.output_size > self.miminum_output_size:
            self.w.contract_out(1, index)
            self.b.contract_out(1, index)
            self.output_size = self.output_size - 1

    def expand_in(self):
        """Add a random input feature"""

        self.w.expand_in(1, self.new_weight_std)
        self.input_size = self.input_size + 1

    def contract_in(self, index):
        """Remove a random input feature"""

        if self.input_size > self.miminum_input_size:
            self.w.contract_in(1, index)
            self.input_size = self.input_size - 1

    def prune(self, n, treshhold=0.001):
        """Remove any features with combined weight values below
        the threshhold
        """
        if self.output_size > self.miminum_output_size:
            if self.w.colsum(n) < treshhold:
                self.contract_out(n)
                return True
        return False

    @property
    def trainable_variables(self):
        """Returns a list of trainable variables"""

        return [self.w.mat, self.b.mat]

    def get_state(self):
        """Returns the current state of the layer"""

        return (
            self.w.get_state(),
            self.b.get_state(),
            self.input_size,
            self.output_size,
        )

    # the given state
    def set_state(self, state):
        """Overwrite the current state of the layer with
        with the given state
        """

        assert not isinstance(state[0], tf.Tensor)
        assert not isinstance(state[1], tf.Tensor)
        self.w.set_state(state[0])
        self.b.set_state(state[1])
        self.input_size = state[2]
        self.output_size = state[3]

    def weight_count(self):
        """Return the number of weights in the layer"""

        return self.input_size * self.output_size + self.output_size

    def summary_string(self):
        return "({}, {})".format(self.input_size, self.output_size)

    def apply_adam(self, gradients, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.w.apply_adam(gradients[0], alpha, beta1, beta2, epsilon)
        self.b.apply_adam(gradients[1], alpha, beta1, beta2, epsilon)

    def __call__(self, inputs):
        """Apply the layer"""

        assert self.w.shape == [self.input_size, self.output_size]
        assert self.b.shape == [1, self.output_size]
        return tf.matmul(inputs, self.w.mat) + self.b.mat


class DynamicConv2DLayer:
    """A convolution layer with dynamic filter size"""

    def __init__(
        self, width, input_size, output_size,
        new_weight_std=0.1,
        miminum_input_size=None,
        miminum_output_size=None
    ):
        """Create the layer with a given initial configuration"""

        if input_size is not None:
            self.w = DynamicMatrix((width, width, input_size, output_size), new_weight_std)
            self.dynamic = True
            self.width = width
            self.input_size = input_size
            self.output_size = output_size
            self.new_weight_std = new_weight_std

            if miminum_input_size is None:
                self.miminum_input_size = input_size
            else:
                self.miminum_input_size = miminum_input_size

            if miminum_output_size is None:
                self.miminum_output_size = output_size
            else:
                self.miminum_output_size = miminum_output_size


    @classmethod
    def from_state(cls, state, new_weight_std=0.01):
        """Initialize from state tuple (or list)"""

        obj = cls(None, None)
        obj.w = DynamicMatrix.from_state(state[0])
        obj.width = state[1]
        obj.input_size = state[2]
        obj.output_size = state[3]
        obj.new_weight_std = new_weight_std
        return obj

    def expand_out(self):
        """Add a random output feature"""

        self.w.expand_out(1, self.new_weight_std)
        self.output_size = self.output_size + 1

    def contract_out(self, n):
        """Remove a random output feature"""

        if self.output_size > self.miminum_output_size:
            self.w.contract_out(1, n)
            self.output_size = self.output_size - 1

    def contract_in(self, n):
        """Remove a random input feature"""

        if self.input_size > self.miminum_input_size:
            self.w.contract_in(1, n)
            self.input_size = self.input_size - 1

    def prune(self, n, treshhold=0.001):
        """Remove any features with combined weight values below
        the threshhold
        """
        if self.output_size > self.miminum_output_size:
            if self.w.colsum(n) < treshhold:
                self.contract_out(n)
                return True
        return False

    def expand_in(self):
        """Add a random input feature"""

        self.w.expand_in(1, self.new_weight_std)
        self.input_size = self.input_size + 1

    @property
    def trainable_variables(self):
        """Returns a list of trainable variables"""
        return [self.w.mat]

    def get_state(self):
        """Returns the current state of the layer"""
        return (self.w.get_state(), self.width, self.input_size, self.output_size)

    # the given state
    def set_state(self, state):
        """Overwrite the current state of the layer with
        the given state
        """

        assert not isinstance(state[0], tf.Tensor)
        self.w.set_state(state[0])
        self.width = state[1]
        self.input_size = state[2]
        self.output_size = state[3]

    def weight_count(self):
        """Return the number of weights in the layer"""
        return self.width * self.width * self.input_size * self.output_size

    def summary_string(self):
        return "({}, {}, {}, {})".format(
            self.width, self.width, self.input_size, self.output_size
        )

    def apply_adam(self, gradients, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.w.apply_adam(gradients[0], alpha, beta1, beta2, epsilon)

    def __call__(self, inputs):
        """Apply the layer"""
        assert self.w.shape == [
            self.width,
            self.width,
            self.input_size,
            self.output_size,
        ]
        return tf.nn.conv2d(inputs, self.w.mat, 2, "SAME")


class DynamicConv2DToDenseLayer:
    """Flattens the output of a conv2d layer and allows
    adding and removing neurons correctly in between
    """

    def __init__(
        self, pixels, features, output_size,
        new_weight_std=0.1,
        miminum_features=None,
        miminum_output_size=None
    ):
        """Create the layer with a given initial configuration"""

        if pixels is not None:
            self.w = DynamicMatrix((pixels * features, output_size), new_weight_std)
            self.b = DynamicMatrix((1, output_size), new_weight_std)
            self.dynamic = True
            self.pixels = pixels
            self.features = features
            self.output_size = output_size
            self.new_weight_std = new_weight_std

            if miminum_features is None:
                self.miminum_features = features
            else:
                self.miminum_features = miminum_features

            if miminum_output_size is None:
                self.miminum_output_size = output_size
            else:
                self.miminum_output_size = miminum_output_size


    @classmethod
    def from_state(cls, state, new_weight_std=0.01):
        """Initialize from state tuple (or list)"""

        obj = cls(None, None)
        obj.w = DynamicMatrix.from_state(state[0])
        obj.b = DynamicMatrix.from_state(state[1])
        obj.features = state[2]
        obj.output_size = state[3]
        obj.new_weight_std = new_weight_std
        return obj

    def expand_out(self):
        """Add a random output feature"""

        self.w.expand_out(1, self.new_weight_std)
        self.b.expand_out(1, self.new_weight_std)
        self.output_size = self.output_size + 1

    def contract_out(self, n):
        """Remove a random output feature"""

        if self.output_size > self.miminum_output_size:
            self.w.contract_out(1, n)
            self.b.contract_out(1, n)
            self.output_size = self.output_size - 1

    def expand_in(self):
        """Add a random input feature"""

        self.w.expand_in(self.pixels, self.new_weight_std)
        self.features = self.features + 1

    def contract_in(self, n):
        """Remove a random input feature"""

        if self.features > self.miminum_features:
            self.w.contract_in(self.pixels, n)
            self.features = self.features - 1

    def prune(self, n, treshhold=0.001):
        """Remove any features with combined weight values below
        the threshhold
        """
        if self.output_size > self.miminum_output_size:
            if self.w.colsum(n) < treshhold:
                self.contract_out(n)
                return True
        return False

    @property
    def trainable_variables(self):
        """Returns a list of trainable variables"""
        return [self.w.mat, self.b.mat]

    def get_state(self):
        """Returns the current state of the layer"""
        return (
            self.w.get_state(),
            self.b.get_state(),
            self.pixels,
            self.features,
            self.output_size,
        )

    def set_state(self, state):
        """Overwrite the current state of the layer with the given state"""

        assert not isinstance(state[0], tf.Tensor)
        assert not isinstance(state[1], tf.Tensor)
        self.w.set_state(state[0])
        self.b.set_state(state[1])
        self.pixels = state[2]
        self.features = state[3]
        self.output_size = state[4]

    def weight_count(self):
        """Return the number of weights in the layer"""
        return self.pixels * self.features * self.output_size + self.output_size

    def summary_string(self):
        return "({}, {}, {})".format(self.pixels, self.features, self.output_size)

    def apply_adam(self, gradients, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.w.apply_adam(gradients[0], alpha, beta1, beta2, epsilon)
        self.b.apply_adam(gradients[1], alpha, beta1, beta2, epsilon)

    def __call__(self, inputs):
        """Apply the layer"""

        assert self.w.shape == [self.pixels * self.features, self.output_size]
        assert self.b.shape == [1, self.output_size]
        # Move pixels to the last columns, so that it is easier to add and remove
        x = tf.transpose(inputs, perm=[0, 3, 1, 2])
        # Now flatten
        x = tf.reshape(x, [x.shape[0], -1])
        x = tf.matmul(x, self.w.mat) + self.b.mat
        return x


class DynamicModel:
    """A model formed of a number of dynamical dense layers"""

    def __init__(self, layers, new_weight_std=0.1, activation=tf.nn.relu):
        """Create the initial model configuration"""

        # A list of layersr in this model
        self.layers = layers

        # Variables related to fully connected part
        self.new_weight_std = new_weight_std
        self.input_size = self.layers[0].input_size
        self.output_size = self.layers[-1].output_size
        self.activation = activation

    def weight_count(self):
        """Returns the number of weights currently in the model"""

        count = 0
        for layer in self.layers:
            if layer.dynamic:
                count += layer.weight_count()
        return count

    def summary(self):
        """Print a summary of the layers in this model"""

        num_weights = 0
        for i, l in enumerate(self.layers):
            if l.dynamic:
                l_weights = l.weight_count()
                num_weights += l_weights
                print(
                    "Layer {}: {},  number of weights {}".format(
                        i, l.summary_string(), l_weights
                    )
                )
        print("Total: {} weights".format(num_weights))

    def expand(self):
        """Add a feature"""

        # Pick a layer
        nl = (int)((len(self.layers) - 1) * np.random.rand())
        l1 = self.layers[nl]
        l2 = self.layers[nl + 1]
        if not l1.dynamic or not l2.dynamic:
            return

        # Expand the number of outputs in the layer
        # and the number of inputs in the next one
        l1.expand_out()
        l2.expand_in()

    def contract(self):
        """Remove a random feature"""

        # Pick a layer
        nl = (int)((len(self.layers) - 1) * np.random.rand())
        l1 = self.layers[nl]
        l2 = self.layers[nl + 1]
        if not l1.dynamic or not l2.dynamic:
            return

        # Choose a random feature
        n = (int)(l1.output_size * np.random.rand())

        # remove it from both the layer and the next one
        l1.contract_out(n)
        l2.contract_in(n)

    def prune(self, treshhold=0.01):
        """Remove any features with combined weight values below
        the threshhold
        """
        for nl in range(len(self.layers)-1):
            l1 = self.layers[nl]
            l2 = self.layers[nl + 1]
            n = 0
            while n < l1.output_size:
                if l1.prune(n, treshhold):
                    l2.contract_in(n)
                else:
                    n += 1

    def stochastic_update(
        self, data, update_function, loss_function, weight_penalty
    ):
        """Stochastic update: change the network and accept the
        change if it decreases the loss function
        """

        # Get the current loss, including the weight penalty
        initial_loss = loss_function(data)
        initial_loss += weight_penalty * self.weight_count()

        # Make note of the current state
        initial_state = self.get_state()

        # Update the network
        update_function()

        # Calculate the loss in the new network
        new_loss = loss_function(data)
        new_loss += weight_penalty * self.weight_count()

        # and the change in the loss
        dloss = new_loss - initial_loss

        # If the loss increases, return to the original state
        if dloss > 0:
            self.set_state(initial_state)
            accepted = False
        else:
            accepted = True

        return accepted

    def stochastic_add_feature(
        self, data, loss_function, weight_penalty=0,
        layer_change_rate=0.1
    ):
        """Stochastic update: add a feature if it decreases
        the loss function
        """
        accepted = self.stochastic_update(
            data, self.expand, loss_function, weight_penalty
        )

        return accepted

    def update_features(
        self, data, loss_function, weight_penalty=0,
        layer_change_rate=0.1
    ):
        """Stochastic update: add or remove a feature if it
        decreases the loss function
        """
        # Randomly choose whether to add or remove
        if np.random.rand() > 0.5:
            update_function = self.expand
        else:
            update_function = self.contract

        accepted = self.stochastic_update(
            data, update_function, loss_function, weight_penalty
        )

        return accepted

    def trainable_variables(self):
        """Returns a list of trainable variables"""

        return [var for layer in self.layers for var in layer.trainable_variables]

    def get_state(self):
        """Returns the current state of the model"""

        state = []
        for layer in self.layers:
            if layer.dynamic:
                state.append(layer.get_state())
        return state

    def set_state(self, state):
        """Overwrite the current state"""

        i = 0
        for layer in self.layers:
            if layer.dynamic:
                layer.set_state(state[i])
                i = i + 1

    def apply_adam(self, gradients, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """Update the weights using the ADAM update method"""
        var_index = 0
        for layer in self.layers:
            n_vars = len(layer.trainable_variables)
            layer.apply_adam(
                gradients[var_index : var_index + n_vars], alpha, beta1, beta2, epsilon
            )
            var_index += n_vars

    def __call__(self, inputs):
        """Apply the model"""

        x = inputs
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation(x)
        x = self.layers[-1](x)
        return x

    # -------------------------------
    # Add or remove dense layers
    # -------------------------------

    def add_layer(self):
        """Add a dense layer.
        The new layer starts close to an identity operation.
        """

        # Pick a layer
        nl = (int)((len(self.layers) - 1) * np.random.rand())
        l1 = self.layers[nl]

        # Build an intermediate layer. Start close to one
        stdiv = self.new_weight_std / (l1.output_size)
        new_w = tf.Variable(
            tf.eye(l1.output_size)
            + tf.random.normal((l1.output_size, l1.output_size), stddev=stdiv),
            trainable=True,
        )
        new_b = tf.Variable(
            tf.random.normal((l1.output_size,), stddev=stdiv), trainable=True
        )
        new_layer = DynamicDenseLayer.from_state(
            (new_w, new_b, l1.output_size, l1.output_size)
        )
        self.layers.insert(nl + 1, new_layer)

    def remove_layer(self):
        """Remove a layer.
        Remove the activation function between two layers and merge
        the now linear operations.
        """

        if len(self.layers) > 2:
            # Pick a layer
            nl = (int)((len(self.layers) - 1) * np.random.rand())

            # Just drop the activation between the layer and the next,
            # reducing them to a single linear operation
            l1 = self.layers[nl]
            l2 = self.layers[nl + 1]

            # Pull the states of the two layers and construct new variables
            st1 = l1.get_state()
            st2 = l2.get_state()
            new_w = tf.Variable(tf.matmul(st1[0], st2[0]), trainable=True)
            new_b = tf.Variable(
                tf.matmul(tf.expand_dims(st1[1], 0), st2[0])[0, :] + st2[1],
                trainable=True,
            )

            assert new_w.shape == (l1.input_size, l2.output_size)

            # Build the new layer
            state = [new_w, new_b, l1.input_size, l2.output_size]
            new_layer = DynamicDenseLayer.from_state(state)

            del self.layers[nl]
            del self.layers[nl]
            self.layers.insert(nl, new_layer)
