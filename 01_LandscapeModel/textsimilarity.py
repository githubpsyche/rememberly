"""
Text Similarity class for computing cosine similarity between text units
in a given reading cycle

Dependencies: sentence_transformers
"""

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

# get device
if torch.cuda.is_available():
    dev = torch.device("cuda:0")
else:
    dev = torch.device("cpu")


class TextSimilarity(nn.Module):
    """
    LSA replacement
    Computes embeddings for a pair of given sentences and calculates the cosine
    similarity between them
    """

    def __init__(self, sbert_model_name="stsb-distilbert-base"):
        """
        sbert_model_name: sbert model to use
        "stsb-roberta-base" is best, but it's double the size of "stsb-distilbert-base",
        with only marginal increase in STS accuracy
        """
        super(TextSimilarity, self).__init__()
        self.model = SentenceTransformer(sbert_model_name)
        self.to(dev)

    @staticmethod
    def cosine_similarity(x):
        """
        Custom implementation of cosine similarities between every text unit
        pdist doesn't make it a square matrix, but we can with pure torch
        returns: cosine similarities matrix of shape (n, n)
        """
        norm = x.norm(dim=-1).unsqueeze(0)

        # this is the formula for cosine similarities in a symmetric matrix
        return x @ x.t() / (norm.t() @ norm)

    def forward(self, reading_cycles):
        """
        Computes cosine similarities between n sentences as a matrix
        designed to take in a list of reading_cycles, not a list of text units
        from the text segmentation process
        reading_cycles: list of lists of text unit strings

        Returns: S, the initial similarity matrix over all text units: shape(num_text_units, num_text_units)
                 embeds, the list of text unit embeddings by SBERT, in case we need them: shape(num_text_units, 768)
        """
        # expand the reading cycles into a list of text units
        text_unit_list = [text_unit for reading_cycle in reading_cycles for text_unit in reading_cycle]

        # embeds is a numpy tensor of text_unit embeddings over all text_units in the reading cycle
        # of shape (n, embed_dim), embed_dim=768 in BERT, n = num_text_units
        embeds = self.model.encode(text_unit_list, convert_to_tensor=True)

        # cosine similarities between every text unit of shape (n, n)
        S = self.cosine_similarity(embeds)

        return S, embeds
