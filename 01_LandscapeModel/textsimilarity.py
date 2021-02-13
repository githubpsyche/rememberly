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

    def forward(self, reading_cycles):
        """
        Computes cosine similarities between n sentences as a matrix
        designed to take in a list of reading_cycles, not a list of text units
        from the text segmentation process
        reading_cycles: list of lists of text unit strings

        Returns: S, the initial similarity matrix over all text units
        """
        # expand the reading cycles into a list of text units
        text_unit_list = [text_unit for reading_cycle in reading_cycles for text_unit in reading_cycle]

        # embeds is the tensor of text_unit embeddings over all text_units in the reading cycle
        # of shape (n, embed_dim), embed_dim=768 in BERT, n = num_text_units
        embeds = self.model.encode(text_unit_list, convert_to_tensor=True)

        # cosine similarities between every text unit of shape (n * (n - 1) // 2, )
        S = 1 - scipy.spatial.distance.pdist(embeds, metric="cosine")

        # return similarity measures and embeddings
        return torch.tensor(S), embeds
