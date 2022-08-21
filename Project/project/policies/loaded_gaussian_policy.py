import numpy as np
import torch
import torch.distributions as dists
from .base_policy import BasePolicy

class Loaded_Gaussian_Policy(BasePolicy):
    def __init__(self, filename, **kwargs):
        super().__init__(**kwargs)
        # get the pre-trained model from the file.
        # Hint: Note that the file contains type of the pre-trained model(discrete or continuous) and the the model.
        checkpoint = torch.load(filename)
        self.discrete = checkpoint['type']
        self.model = checkpoint['model']
        if not self.discrete:
            self.logstd = checkpoint['logstd']

    def update(self, obs_no, acs_na, adv_n=None, acs_labels_na=None):
        print("\n\nThis policy class simply loads in a particular type of policy and queries it.")
        print("Not training procedure has been written, so do not try to train it.\n\n")
        raise NotImplementedError

    def get_action(self, obs):
        # get the actions from observations.
        # Hint: You've implemented this before.
        obs = torch.tensor(obs, dtype=torch.float)
        with torch.no_grad():
            if self.discrete:
                logits = self.model(obs)
                acs = torch.distributions.Categorical(torch.nn.Softmax(dim=1)(logits)).sample()
            else:
                means = self.model(obs)
                acs = means + torch.mul(torch.exp(self.logstd), torch.randn(len(obs), len(means[0])))
        return acs.numpy()