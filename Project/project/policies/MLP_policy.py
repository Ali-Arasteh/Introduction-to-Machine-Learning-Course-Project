import numpy as np
import torch
import torch.distributions as dists
from .base_policy import BasePolicy
from project.infrastructure.torch_utils import build_mlp

class MLPPolicy(BasePolicy):
    def __init__(self, ac_dim, ob_dim, n_layers, size, learning_rate=1e-4, training=True, discrete=False, **kwargs):
        super().__init__(**kwargs)
        # init vars
        self.discrete = discrete
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.size = size
        self.learning_rate = learning_rate
        self.training = training
        # Building the graph
        self.build_graph()

    def build_graph(self):
        self.define_forward_pass()
        if self.training:
            self.define_train()

    def define_forward_pass(self):
        if self.discrete:
            self.model = build_mlp(self.ob_dim, output_size=self.ac_dim, n_layers=self.n_layers, size=self.size)
        else:
            self.model = build_mlp(self.ob_dim, output_size=self.ac_dim, n_layers=self.n_layers, size=self.size)
            self.logstd = torch.tensor(torch.zeros(self.ac_dim), requires_grad=True)

    def define_train(self):
        raise NotImplementedError

    def save(self, filepath):
        if self.discrete:
            torch.save({'type': self.discrete, 'model': self.model}, filepath)
        else:
            torch.save({'type': self.discrete, 'model': self.model, 'logstd': self.logstd}, filepath)

    def restore(self, filepath):
        raise NotImplementedError

    # query this policy with observation(s) to get selected action(s)
    def get_action(self, obs):
        # Use the model described before to get the actions from observations.
        # Hint1: Use torch.no_grad() in order to not affect the training procedure.
        # Hint2: You should separately consider the case of the discrete problem and the continuous problem.
        # Hint3: Take a look at torch.distributions.Categorical and torch.randn
        obs = torch.tensor(obs, dtype=torch.float)
        with torch.no_grad():
            if self.discrete:
                logits = self.model(obs)
                acs = torch.distributions.Categorical(torch.nn.Softmax(dim=1)(logits)).sample()
            else:
                means = self.model(obs)
                acs = means + torch.mul(torch.exp(self.logstd), torch.randn(len(obs), self.ac_dim))
        return acs.numpy()

    # update/train this policy
    def update(self, observations, actions):
        raise NotImplementedError

class MLPPolicySL(MLPPolicy):
    """
        This class is a special case of MLPPolicy,
        which is trained using "Supervised learning".
        The relevant functions to define are included below.
    """
    def define_train(self):
        # Define the Adam optimizer in both case (discrete and continuous) and save it as self.optimizer.
        # define what exactly the optimizer should minimize when updating the policy. (the loss function)
        #  and save as self.loss_fn
        # Hint1: You should separately consider the case of the discrete problem and the continuous problem.
        # Hint2: Look up torch.optim.Adam
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        if self.discrete:
            self.loss_fn = torch.nn.CrossEntropyLoss()
        else:
            self.loss_fn = torch.nn.MSELoss()

    def update(self, observations, actions):
        """
            #inputs:
                observations: the list of the acquired observations from environment.
                actions: the list of the actions taken by the expert for the aforementioned observations.

            #outputs:
                loss: value of loss function in this step.
        """
        assert self.training, 'Policy must be created with training=True in order to perform training updates...'
        # Get the action prediction from the model.
        # Hint: You've implemented this before!
        observations = torch.tensor(observations, dtype=torch.float)
        if self.discrete:
            logits = self.model(observations)
            acts_prediction = torch.nn.Softmax(dim=1)(logits)
            loss = self.loss_fn(acts_prediction, torch.tensor(actions, dtype=torch.long))
        else:
            means = self.model(observations)
            acts_prediction = means + torch.mul(torch.exp(self.logstd), torch.randn(len(observations), self.ac_dim))
            loss = self.loss_fn(acts_prediction, torch.tensor(actions))
        # Use the loss to go one step forward in training.
        # Hint: Be careful! use optimizer.zero_grad() before the backward algorithm.
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        if not self.discrete:
            with torch.no_grad():
                self.logstd -= self.learning_rate * self.logstd.grad
                self.logstd.grad = None
        return loss.detach().numpy()