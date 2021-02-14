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


class Landscape():
    """
    Human memory model for text comprehension
    Combines text similarity measure and LS-R algorithm

    Parameters to fit:
      maximum_activation
      decay_rate
      working_memory_capacity
      lambda_lr
      semantic_strength_coeff
      SBERT() also possible
    can add semantic_strength_coeff if necessary
    """

    def __init__(self, sbert_model_name,
                 initial_max_activation=1.0,
                 initial_decay_rate=0.1,
                 initial_memory_capacity=5.0,
                 initial_learning_rate=0.9,
                 initial_semantic_strength_coeff=1.0):

        """
        reading_cycles is list of lists input to the TextSimilarity measure
        initial_similarities is the output of a TextSimilarity over reading_cycles
        """
        super(Landscape, self).__init__()

        # create the text similarity measure
        self.sbert_model = sbert_model_name
        # torch.jit.trace this
        self.text_similarity = TextSimilarity(sbert_model_name)

        # initialize trainable parameters
        self.maximum_activation = initial_max_activation
        self.decay_rate = initial_decay_rate
        self.working_memory_capacity = initial_memory_capacity
        self.lambda_lr = initial_learning_rate
        self.semantic_strength_coeff = initial_semantic_strength_coeff

        # non-negotiable minimum activation
        self.minimum_activation = 0

        # various variables
        self.reading_cycles = None
        self.reading_cycle_lengths = None
        self.activations = None
        self.S = None
        self.cycle_idx = 0
        self.history = None
        self.recalling = True
        self.recall_cycles = None
        self.recall_idx = 0

    @staticmethod
    def sigma(x):
        """
        Sigma function for positive logarithmic connection strength in S
        as per Yeari, van den Broek, 2016
        replace with simple sigmoid?
        """
        return np.tanh(3 * (x - 1)) + 1

    def update_activations(self, num_prev_text_units, cycle_len):
        """
        Updates input activations for a single reading cycle, given
          activations: from previous cycle
          S: similarity matrix from previous cycle
          num_prev_text_units: the number of previously read units in the set of reading cycles
          cycle_len: the number of text units in current cycle
        """
        if self.activations is None or self.S is None:
            return Warning("store reading_cycles with __call__")

        self.activations = self.decay_rate * (self.sigma(self.S) @ self.activations.T).T

        # working memory simulation
        activation_sum = self.activations.sum() + 1e-6
        if activation_sum > self.working_memory_capacity:
            # scale activations proportionally so their sum equals working memory capacity
            self.activations = self.activations * self.working_memory_capacity / activation_sum

        # attention simulation: set current reading cycle activations to max_val
        self.activations[:, num_prev_text_units - cycle_len:num_prev_text_units] = (
                np.ones((1, cycle_len)) * self.maximum_activation
        )
        return self.activations + 1e-6

    def update_S(self):
        """
        Update S matrix for a single reading cycle
        """
        if self.activations is None or self.S is None:
            raise Warning("store reading_cycles with __call__")
        self.S = self.S + self.lambda_lr * self.activations.T @ self.activations
        self.S = self.semantic_strength_coeff * self.S
        return self.S

    def cycle(self):
        """
        Complete update self for a single reading cycle, given the parameters necessary to update
        the activations
        """
        if self.recalling:
            raise Warning("recalling")

        num_prev_text_units = sum(self.reading_cycle_lengths[:self.cycle_idx + 1])
        cycle_len = self.reading_cycle_lengths[self.cycle_idx]

        self.activations = self.update_activations(num_prev_text_units, cycle_len)
        self.S = self.update_S()
        self.cycle_idx += 1
        if self.cycle_idx == 1:
            self.history = self.activations.copy()
        else:
            self.history = np.concatenate([self.history, self.activations.copy()], axis=0)

        return self.activations, self.S

    def output_probabilities(self, sensitivity=1.0):
        """
        return current model state as probabilities
        """
        act_logits = np.power(self.activations, sensitivity)
        act_probs = act_logits / (act_logits.sum() + 1e-6)
        hist_logits = np.power(self.history, sensitivity)
        hist_probs = hist_logits / (np.expand_dims(hist_logits.sum(axis=-1), axis=0) + 1e-6)

        return act_probs, hist_probs

    def get_activations(self):
        return self.activations.copy()

    def get_S(self):
        return self.S.copy()

    def get_history(self):
        return self.history.copy()

    def force_recall(self, activation_idx, sensitivity=1.0):
        """
        Force recall: update activations but not connections
        """
        # find index and create a new cycle
        self.recalling = True
        self.activations = self.update_activations(activation_idx + 1, 1)
        return self.output_probabilities(sensitivity)[0]

    def __call__(self, reading_cycles):
        """
        Essentially a reset button. This ensures that you can use the same model on
        multiple sets of reading cycles
        Store reading_cycles and initialize/reinitialize the activations to 0
        compute the initial similarities matrix
        """
        self.recalling = False
        self.reading_cycles = reading_cycles[:]
        self.reading_cycle_lengths = []
        for reading_cycle in reading_cycles:
            self.reading_cycle_lengths.append(len(reading_cycle))
        self.reading_cycle_lengths += [0]
        self.recall_cycles = []
        for reading_cycle in reading_cycles:
            self.recall_cycles += [*reading_cycle]

        self.activations = np.zeros((1, sum(self.reading_cycle_lengths)))

        # intialize similarities matrix to input; no need for cloning, I think
        self.S, _ = self.text_similarity(reading_cycles)
        self.history = self.activations.copy()
        self.cycle_idx = 0
        self.recall_idx = 0
