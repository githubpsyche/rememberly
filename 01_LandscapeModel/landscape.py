"""
Initial implementation of Landscape model
To initialize it, pass in the
"""

import torch
from torch import nn

# get device
if torch.cuda.is_available():
    dev = torch.device("cuda:0")
else:
    dev = torch.device("cpu")


class Landscape(nn.Module):
    """
    Human memory model for text comprehension

    Parameters to fit:
      maximum_activation
      decay_rate
      working_memory_capacity
      lambda_lr
    can add semantic_strength_coeff if necessary
    """

    def __init__(self, reading_cycles,
                 initial_similarities,
                 initial_max_activation=1.0,
                 initial_decay_rate=0.1,
                 initial_memory_capacity=5.0,
                 initial_learning_rate=0.9):
        """
        reading_cycles is list of lists input to the TextSimilarity measure
        initial_similarities is the output of a TextSimilarity over reading_cycles
        """
        super(Landscape, self).__init__()
        # to be able to reset the model
        self.initial_params = (
            reading_cycles, initial_similarities, initial_max_activation,
            initial_decay_rate, initial_memory_capacity, initial_learning_rate
        )
        # initialize trainable parameters
        # what's a good initial max activation?
        self.maximum_activation = nn.Parameter(
            torch.FloatTensor([initial_max_activation])
        )
        self.decay_rate = nn.Parameter(
            torch.FloatTensor([initial_decay_rate])
        )
        self.working_memory_capacity = nn.Parameter(
            torch.FloatTensor([initial_memory_capacity])
        )
        self.lambda_lr = nn.Parameter(
            torch.FloatTensor([initial_learning_rate])
        )

        # non-negotiable minimum activation
        self.minimum_activation = 0

        # to be able to use a vector and determine when to stop "reading" on the fly
        self.reading_cycle_lengths = [len(reading_cycle) for reading_cycle in reading_cycles] + [0]

        # single vector to make calculations much simpler
        # randomly initialized between [0, 1] -> is this a good idea?
        self.activations = torch.zeros(1, sum(self.reading_cycle_lengths),
                                       requires_grad=True, device=dev)

        # intialize similarities matrix to input; no need for cloning, I think
        self.S = initial_similarities.clone().to(dev)

        # to prevent calling forward() more than once
        self.forward_calls = 0

        self.to(dev)

    @staticmethod
    def sigma(x):
        """
        Sigma function for positive logarithmic connection strength in S
        as per Yeari, van den Broek, 2016
        replace with simple sigmoid?
        """
        return torch.tanh(3 * (x - 1)) + 1

    def reset(self, *params):
        """
        Reset so that we don't have to create new instances for every set of reading cycles
        """
        if len(params) == 1:
            raise ValueError("If using new reading cycles, also input the new initial similarity matrix")
        params_ = [*params, *self.initial_params[len(params):]]
        self.__init__(*params_)

    def update_activations(self, num_prev_text_units, cycle_len):
        """
        For customizability
        Updates activations for a single reading cycle, given
          num_prev_text_units: the number of previously read units in the set of reading cycles
          cycle_len: the number of text units in current cycle
        """
        self.activations = self.decay_rate * (self.sigma(self.S) @ self.activations.t()).t()

        # working memory simulation
        activation_sum = self.activations.sum()
        if activation_sum > self.working_memory_capacity:
            # scale activations proportionally so their sum equals working memory capacity
            self.activations = self.activations * self.working_memory_capacity / activation_sum

        # attention simulation: set current reading cycle activations to max_val
        self.activations[:, num_prev_text_units - cycle_len:num_prev_text_units] = (
                torch.ones(1, cycle_len, device=dev) * self.maximum_activation
        )

    def update_S(self):
        """
        For customizability
        Updates S matrix for a single reading cycle
        """
        self.S = self.S + self.lambda_lr * self.activations.t() @ self.activations

    def cycle(self, num_prev_text_units, cycle_len):
        """
        Complete update self for a single reading cycle, given the parameters necessary to update
        the activations
        """
        self.update_activations(num_prev_text_units, cycle_len)
        self.update_S()

    def forward(self):
        """
        Compute the entire activation and update process

        PERFORM THIS ONLY ONCE
        OTHERWISE THE S matrix and activations will keep updating and growing larger and larger
        """
        if self.forward_calls > 0:
            raise Warning("forward() more than once")
        self.forward_calls += 1

        num_text_units = 0

        # for each reading cycle
        for rc_len in self.reading_cycle_lengths:
            # get the number of prior text units
            num_text_units += rc_len
            self.cycle(num_text_units, rc_len)

        # just to see the output
        return self.activations, self.S
