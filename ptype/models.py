from tensorflow.keras.layers import (
    Dense,
    Dropout,
    GaussianNoise,
    LeakyReLU,
)
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.regularizers import l1, l2, l1_l2
import tensorflow as tf
import numpy as np
import logging
import sys

from imblearn.under_sampling import RandomUnderSampler
from imblearn.tensorflow import balanced_batch_generator
from evml.keras.losses import DirichletEvidentialLoss
from evml.keras.callbacks import ReportEpoch


logger = logging.getLogger(__name__)


class DenseNeuralNetwork(object):
    """
    A Dense Neural Network Model that can support arbitrary numbers of hidden layers.
    Attributes:
        hidden_layers: Number of hidden layers
        hidden_neurons: Number of neurons in each hidden layer
        activation: Type of activation function
        output_activation: Activation function applied to the output layer
        optimizer: Name of optimizer or optimizer object.
        loss: Name of loss functions or loss objects (can match up to number of output layers)
        loss_weights: Weights to be assigned to respective loss/output layer
        use_noise: Whether or not additive Gaussian noise layers are included in the network
        noise_sd: The standard deviation of the Gaussian noise layers
        lr: Learning rate for optimizer
        use_dropout: Whether or not Dropout layers are added to the network
        dropout_alpha: proportion of neurons randomly set to 0.
        batch_size: Number of examples per batch
        epochs: Number of epochs to train
        l2_weight: L2 weight parameter
        sgd_momentum: SGD optimizer momentum parameter
        adam_beta_1: Adam optimizer beta_1 parameter
        adam_beta_2: Adam optimizer beta_2 parameter
        decay: Level of decay to apply to learning rate
        verbose: Level of detail to provide during training (0 = None, 1 = Minimal, 2 = All)
        classifier: (boolean) If training on classes
    """

    def __init__(
        self,
        hidden_layers=1,
        hidden_neurons=4,
        activation="relu",
        output_activation="softmax",
        optimizer="adam",
        loss="categorical_crossentropy",
        loss_weights=None,
        annealing_coeff=None,
        use_noise=False,
        noise_sd=0.0,
        lr=0.001,
        use_dropout=False,
        dropout_alpha=0.2,
        batch_size=128,
        epochs=2,
        kernel_reg="l2",
        l1_weight=0.0,
        l2_weight=0.0,
        sgd_momentum=0.9,
        adam_beta_1=0.9,
        adam_beta_2=0.999,
        epsilon=1e-7,
        decay=0,
        verbose=0,
        classifier=False,
        random_state=1000,
        callbacks=[],
        balanced_classes=0,
        steps_per_epoch=0,
    ):

        self.hidden_layers = hidden_layers
        self.hidden_neurons = hidden_neurons
        self.activation = activation
        self.output_activation = output_activation
        self.optimizer = optimizer
        self.optimizer_obj = None
        self.sgd_momentum = sgd_momentum
        self.adam_beta_1 = adam_beta_1
        self.adam_beta_2 = adam_beta_2
        self.epsilon = epsilon
        self.loss = loss
        self.loss_weights = loss_weights
        self.annealing_coeff = annealing_coeff
        self.lr = lr
        self.kernel_reg = kernel_reg
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.batch_size = batch_size
        self.use_noise = use_noise
        self.noise_sd = noise_sd
        self.use_dropout = use_dropout
        self.dropout_alpha = dropout_alpha
        self.epochs = epochs
        self.callbacks = callbacks
        self.decay = decay
        self.verbose = verbose
        self.classifier = classifier
        self.y_labels = None
        self.model = None
        self.random_state = random_state
        self.balanced_classes = balanced_classes
        self.steps_per_epoch = steps_per_epoch

    def build_neural_network(self, inputs, outputs):
        """
        Create Keras neural network model and compile it.
        Args:
            inputs (int): Number of input predictor variables
            outputs (int): Number of output predictor variables
        """
        if self.activation == "leaky":
            self.activation = LeakyReLU()

        if self.kernel_reg == "l1":
            self.kernel_reg = l1(self.l1_weight)
        elif self.kernel_reg == "l2":
            self.kernel_reg = l2(self.l2_weight)
        elif self.kernel_reg == "l1_l2":
            self.kernel_reg = l1_l2(self.l1_weight, self.l2_weight)
        else:
            self.kernel_reg = None

        self.model = tf.keras.models.Sequential()
        self.model.add(
            Dense(
                inputs,
                activation=self.activation,
                kernel_regularizer=self.kernel_reg,
                name="dense_input",
            )
        )

        for h in range(self.hidden_layers):
            self.model.add(
                Dense(
                    self.hidden_neurons,
                    activation=self.activation,
                    kernel_regularizer=self.kernel_reg,
                    name=f"dense_{h:02d}",
                )
            )
            if self.use_dropout:
                self.model.add(Dropout(self.dropout_alpha, name=f"dropout_{h:02d}"))
            if self.use_noise:
                self.model.add(GaussianNoise(self.noise_sd, name=f"noise_{h:02d}"))

        self.model.add(
            Dense(outputs, activation=self.output_activation, name="dense_output")
        )

        if self.optimizer == "adam":
            self.optimizer_obj = Adam(
                learning_rate=self.lr,
                beta_1=self.adam_beta_1,
                beta_2=self.adam_beta_2,
                epsilon=self.epsilon,
                #decay_rate=self.decay,
            )
        elif self.optimizer == "sgd":
            self.optimizer_obj = SGD(
                learning_rate=self.lr, momentum=self.sgd_momentum, #decay_rate=self.decay
            )

        self.model.build((self.batch_size, inputs))
        self.model.compile(
            optimizer=self.optimizer_obj,
            loss=self.loss
            #             metrics = ["accuracy",
            #                tf.keras.metrics.Precision(name="prec"),
            #                tf.keras.metrics.Recall(name="recall"),
            #                tfa.metrics.F1Score(num_classes=4,
            #                                    average='macro', name="f1"),
            #                tf.keras.metrics.AUC(name = "auc")]
        )
        # print(self.model.summary())

    def fit(self, x_train, y_train, validation_data=None):

        inputs = x_train.shape[-1]
        outputs = y_train.shape[-1]

        if self.loss == "dirichlet":
            for callback in self.callbacks:
                if isinstance(callback, ReportEpoch):
                    # Don't use weights within Dirichelt, it is done below using sample weight
                    self.loss = DirichletEvidentialLoss(
                        callback=callback, name=self.loss
                    )
                    break
            else:
                raise OSError(
                    "The ReportEpoch callback needs to be used in order to run the evidential model."
                )
        self.build_neural_network(inputs, outputs)
        if self.balanced_classes:
            train_idx = np.argmax(y_train, 1)
            training_generator, steps_per_epoch = balanced_batch_generator(
                x_train,
                y_train,
                sample_weight=np.array([self.loss_weights[_] for _ in train_idx]),
                sampler=RandomUnderSampler(),
                batch_size=self.batch_size,
                random_state=self.random_state,
            )
            history = self.model.fit(
                training_generator,
                validation_data=validation_data,
                steps_per_epoch=steps_per_epoch,
                batch_size=self.batch_size,
                epochs=self.epochs,
                verbose=self.verbose,
                callbacks=self.callbacks,
                shuffle=True,
            )
        else:
            sample_weight = np.array([self.loss_weights[np.argmax(_)] for _ in y_train])
            if not self.steps_per_epoch:
                self.steps_per_epoch = sample_weight.shape[0] // self.batch_size
            history = self.model.fit(
                x=x_train,
                y=y_train,
                validation_data=validation_data,
                batch_size=self.batch_size,
                epochs=self.epochs,
                verbose=self.verbose,
                callbacks=self.callbacks,
                sample_weight=sample_weight,
                steps_per_epoch=self.steps_per_epoch,
                # class_weight={k: v for k, v in enumerate(self.loss_weights)},
                shuffle=True,
            )
        return history

    def predict(self, x):
        y_prob = self.model.predict(x, batch_size=self.batch_size, verbose=self.verbose)
        return y_prob

    def predict_dropout(self, x, mc_forward_passes=10):
        y_prob = np.stack(
            [
                np.vstack(
                    [
                        self.model(tf.expand_dims(lx, axis=-1), training=True)
                        for lx in np.array_split(x, x.shape[0] // self.batch_size)
                    ]
                )
                for _ in range(mc_forward_passes)
            ]
        )
        pred_probs = y_prob.mean(axis=0)
        epistemic_variance = y_prob.var(axis=0)
        # Calculating entropy across multiple MCD forward passes
        epsilon = sys.float_info.min
        entropy = -np.sum(
            pred_probs * np.log(pred_probs + epsilon), axis=-1
        )  # shape (n_samples,)
        # Calculating mutual information across multiple MCD forward passes
        mutual_info = entropy - np.mean(
            np.sum(-y_prob * np.log(y_prob + epsilon), axis=-1), axis=0
        )  # shape (n_samples,)
        return pred_probs, epistemic_variance, entropy, mutual_info

    def predict_proba(self, x):
        y_prob = self.model.predict(x, batch_size=self.batch_size, verbose=self.verbose)
        return y_prob
