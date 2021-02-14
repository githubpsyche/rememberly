Raw data as originally collected for collaborations with the Brown-Schmidt and Fazio labs are stored in `raw/`.

Text data are pooled from `raw/` into a single hierarchy of source texts and recall data with a common naming and formatting scheme within `texts/`. The code that carries out the transformation is in `pipeline/data-preprocessing/PoolTextData.ipynb`.

Human-coded sequencing of text data are organized with the same scheme into `sequences/human` using `pipeline/data-preprocessing/PoolSequenceData.ipynb`; machine-coded sequences are given a parallel organization in `sequences/machine`.

Intermediate outputs obtained during sentence simplification, proposition extraction, coreference resolution, and semantic representation are organized into `simple_sentences`, `propositions`, `coreferences`, and `embeddings`, respectively.