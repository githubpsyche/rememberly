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
    Combines text similarity measure and LS-R algorithm

    Parameters to fit:
      maximum_activation
      decay_rate
      working_memory_capacity
      lambda_lr
      SBERT() (optional)
    can add semantic_strength_coeff if necessary
    """

    def __init__(self, sbert_model_name,
                 initial_max_activation=1.0,
                 initial_decay_rate=0.1,
                 initial_memory_capacity=5.0,
                 initial_learning_rate=0.9):

        """
        reading_cycles is list of lists input to the TextSimilarity measure
        initial_similarities is the output of a TextSimilarity over reading_cycles
        """
        super(Landscape, self).__init__()

        # create the text similarity measure
        self.sbert_model = sbert_model_name
        self.text_similarity = TextSimilarity(sbert_model_name)

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

        # shift to device
        self.to(dev)
        self.device = dev

    @staticmethod
    def sigma(x):
        """
        Sigma function for positive logarithmic connection strength in S
        as per Yeari, van den Broek, 2016
        replace with simple sigmoid?
        """
        return torch.tanh(3 * (x - 1)) + 1

    def update_activations(self, activations, S, num_prev_text_units, cycle_len):
        """
        Updates input activations for a single reading cycle, given
          activations: from previous cycle
          S: similarity matrix from previous cycle
          num_prev_text_units: the number of previously read units in the set of reading cycles
          cycle_len: the number of text units in current cycle
        """
        activations = self.decay_rate * (self.sigma(S) @ activations.t()).t()

        # working memory simulation
        activation_sum = activations.sum() + 1e-6
        if activation_sum > self.working_memory_capacity:
            # scale activations proportionally so their sum equals working memory capacity
            activations = activations * self.working_memory_capacity / activation_sum

        # attention simulation: set current reading cycle activations to max_val
        activations[:, num_prev_text_units - cycle_len:num_prev_text_units] = (
                torch.ones(1, cycle_len, device=dev) * self.maximum_activation
        )
        return activations

    def update_S(self, activations, S):
        """
        Update S matrix for a single reading cycle
        """
        S = S + self.lambda_lr * activations.t() @ activations
        return S

    def cycle(self, activations, S, num_prev_text_units, cycle_len):
        """
        Complete update self for a single reading cycle, given the parameters necessary to update
        the activations
        """
        activations = self.update_activations(activations, S, num_prev_text_units, cycle_len)
        S = self.update_S(activations, S)
        return activations, S

    def forward(self, reading_cycles, return_as_cycles=False):
        """
        Compute the entire activation and update process for the input reading cycles
        """
        reading_cycle_lengths = [len(reading_cycle) for reading_cycle in reading_cycles] + [0]

        # initialize activations to a single vector to make calculations much simpler
        activations = torch.zeros(1, sum(reading_cycle_lengths),
                                  requires_grad=True, device=self.device)

        num_text_units = 0

        # intialize similarities matrix to input; no need for cloning, I think
        S, _ = self.text_similarity(reading_cycles)
        S = S.to(self.device)

        # initialize empty history list for output visualization
        # to track evolution of activations
        history = []

        # for each reading cycle
        for rc_len in reading_cycle_lengths:
            # get the number of prior text units
            num_text_units += rc_len
            # update activations and S
            activations, S = self.cycle(activations, S, num_text_units, rc_len)
            # append to history
            history.append(activations)

        # concat history to a matrix - (num_reading_cycles + 1, num_text_units)
        # i.e., it is a cycles by text_units matrix
        # do we need the + 1 -> I put it to simulate finishing reading the cycles
        history = torch.cat(history, dim=0)

        # output
        if return_as_cycles:
            out = []
            prev_idx = 0
            for rc_len in reading_cycle_lengths[:-1]:
                out += [activations[:, prev_idx:prev_idx + rc_len]]
                prev_idx += rc_len
            return out, S, history

        return activations, S, history

    def output_probabilities(self, reading_cycles, sensitivity=1.0,
                             return_as_cycles=False):
        """
        Extension of forward to get activations as a distribution vector, or as a
        set of distribution vectors for each cycle
        """
        reading_cycle_lengths = [len(reading_cycle) for reading_cycle in reading_cycles]

        activations, _, history = self(reading_cycle, return_as_cycles=False)
        act_logits = torch.pow(activations, sensitivity)
        hist_logits = torch.pow(history, sensitivity)

        # softmax is problematic - not outputting clean distributions
        act_probs = act_logits / (act_logits.sum() + 1e-6)
        hist_probs = hist_logits / (hist_logits.sum(dim=-1).unsqueeze(1) + 1e-6)

        if return_as_cycles:
            out = []
            hist_out = {}
            # initialize all history cycle outputs to an empty list
            for i in range(hist_probs.shape[0]):
                hist_out[f"cycle_{i}"] = []

            prev_idx = 0
            count = 0
            for rc_len in reading_cycle_lengths:
                out += [act_probs[:, prev_idx:prev_idx + rc_len]]

                for i in range(hist_probs.shape[0]):
                    hist_out[f"cycle_{i}"].append(hist_probs[i, prev_idx:prev_idx + rc_len])

                prev_idx += rc_len

            return out, hist_out

        return act_probs, hist_probs
