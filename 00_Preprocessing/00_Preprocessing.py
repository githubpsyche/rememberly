# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.7.1
# ---

# %% [markdown]
# # Text Preprocessing
# We have a lot of options for preprocessing text before input into the model. For this project, the main priority is to tokenize simple sentences and propositions so that 1) they can serve as units in the model, and 2) so that small orthographical (but not semantic) differences between these units don't distort the similarity structure between units.
#
# We'll thus include the following steps in our preprocessing pipeline:
# 1. **Lowercase all tokens.** Sometimes case matters, and can be an important cue for tokenization itself, but before computing semantic similiarities, we'll remove it as a factor in our calculations.
# 2. **Divide text into sentences and into simple propositions**. As the project scales, we want to model people's representations of assertions encoded into a passage, not just words. Therefore, even propositional tokenization is an unsolved problem in NLP compared to sentence and word tokenization, we'll use it to specify units in our model. Sentences will define cycles.
# 3. **Keep most punctuation.** Word-based NLP models tend to discard punctuation from their corpuses and representations, but sentence similarity tools use punctuation just as well as word order to make accurate predictions. We will however discard line breaks

# %% [markdown]
# ## Segmentation
# We preserve case and context while performing segmentation, so it comes first. We'll initially rely on spacy to segment passages by sentences, but could switch to GCP to make an appeal for the prize with only small changes.

# %% [markdown]
# ### Sentence Segmentation
# Proposition segmentation requires sentence segmentation first so we'll do that.

# %%
# export
import spacy

nlp = spacy.load("en_core_web_sm")

def sentences(passage):
    """
    Breaks input passage into describe sentences. Lines with just punctuation or stop characters can be returned, but should only be used to ensure consistent passage structure in downstream renders.
    """
    return [each.text for each in nlp(passage).sents]

# %%
example = """It was a hot, sunny day and Kaylie and Rachel decided to take advantage of their off day and take a trip to the beach. The sun was an optimistic yellow pellet, blistering in the sky as they were on their way to Old Orchard Beach in Maine. Once arriving and walking from the parking lot, they spotted the crystal clear water that faded into a deep blue on the horizon. The beach was swarming with people. Some on yellow, orange, white, and pink towels, while others were seen stretched out on lounge chairs with broad-brimmed hats shading their eyes from the rays of the sun. Children were decorating sandcastles with smooth, round seashells, and Kaylie decided to stretch out on her towel in the sand for a quick nap while Rachel went to cool off in the ocean. 
The sun shined on the water, causing it to look like a million little crystals. Just before making it to the surf, Rachel saw a little girl with freckles in a light purple bathing suit and little yellow floaties on her arms. The girl darted in front of Rachel, closely followed by a boy in green swimming shorts. Rachel smiled as she reminisced on memories of beach days with her brothers. After spending some time in the water, she looked up and saw four teenagers flying colorful paragliders. She thought to herself that the day was truly perfect, and all anxieties from her daily life were absent for a while. She closed her eyes for a few seconds to savor the moment before going to lie down next to Kaylie in the warm sand."""

sentences(example)

# %% [markdown]
# ### Proposition Segmentation
# Finding a purely Python-based implementation of OIE that doesn't conflict with other dependencies we're likely to have (e.g Pytorch) has been pretty tough. ClausIE is one of the most well-known OIE tools, and `claucy` claims to be a spacy-based reproduction of it. Installing it requires cloning the underlying repository and doing an editable install, though! I hope this doesn't create issues.

# %%
# export
import claucy

claucy.add_to_pipe(nlp)

# %%
# export 
def propositions(sentence):
    """
    Returns the list of simple propositions detected by the open information extraction tool ClausIE.
    """
    doc = nlp(sentence)
    return [each.to_propositions(as_text=True) for each in doc._.clauses]

# %%
example_index = 4
print(sentences(example)[example_index])
propositions(sentences(example)[example_index])
# %% [markdown]
# Interestingly, this specific OIE tool returns multiple representations for each clause structure. We can pick some heuristic for selecting between them across applications - maybe the longest version for semantic association build, and the shortest when we need to summarize:

# %%

def long_propositions(sentence):
    """
    Heuristic for selecting between proposition representations that always prefers the longest string and removes case.
    """
    initial_propositions = propositions(sentence)
    props = []
    for proposition_list in initial_propositions:
        choice = proposition_list[0]
        for p in proposition_list:
            if len(p) < len(choice):
                choice = p
        props.append(choice.lower())
    return props

# %% [markdown]
# # Cycles and Units
# We want to generate for any arbitrary text a list of sentences and inside that of included propositions to support model simulation and beyond.

# %%

def cycles_and_units(passage):
    """
    Generate for any arbitrary text a list of sentences and inside that of included propositions to support model simulation and beyond.
    """
    return (sentences(passage),
    [long_propositions(sentence) for sentence in sentences(passage)])
