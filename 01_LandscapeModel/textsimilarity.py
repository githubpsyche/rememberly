"""
Text Similarity class for computing cosine similarity between text units
in a given reading cycle

Dependencies: sentence_transformers
"""

import torch
import torch.nn.functional as F
from scipy.spatial import distance
from sentence_transformers import SentenceTransformer

# get device
if torch.cuda.is_available():
    dev = torch.device("cuda:0")
else:
    dev = torch.device("cpu")


class TextSimilarity(torch.nn.Module):
    """
    Computes embeddings for a pair of given sentences and calculates the cosine
    similarity between them
    """
    def __init__(self, sbert_model_name):
        """
        sbert_model_name: sbert model to use, I think "stsb-roberta-base" is good
        """
        super(TextSimilarity, self).__init__()
        self.model = SentenceTransformer(sbert_model_name)
        self.to(dev)

    @staticmethod
    def measure(embedded_reading_cycle):
        """
        Computes cosine similarity between all text unit embeddings in a reading cycle
        reading_cycle: list of text unit embeddings
        """
        # no customizable metric - only cosine, otherwise 1 - x doesn't make sense
        similarities = 1 - scipy.spatial.distance.pdist(embedded_reading_cycle, metric="cosine")
        # convert to torch tensors
        return torch.tensor(similarities)

    def forward(self, reading_cycles):
        """
        Computes cosine similarities between n sentences as a matrix
        reading_cycles: list of lists of text unit strings
        """
        # embeds is the list of text_unit embeddings for each reading cycle of shape
        # (num_reading_cycles, num_text_units, embed_dim)
        # where num_text_units varies per reading cycle, and embed_dim = 768 as per BERT
        embeds = [self.model.encode(reading_cycle, convert_to_tensor=True) for reading_cycle in reading_cycles]
        # given n text units in a given reading_cycle
        # pdist will compute n * (n - 1) // 2 similarities, so measures is a list of shape
        # of shape (num_reading_cycles, num_text_units * (num_text_units - 1) // 2)
        measures = [self.measure(emb_reading_cycle) for emb_reading_cycle in embeds]
        # return similarity measures and embeddings
        return measures, embeds
