# Original notebook provenance

The user-supplied `Spam Detector.ipynb` contains 23 code cells and used the local CSV path `/home/jovyan/Machine Learning/proyecto/spam.csv`. It counted 5,572 messages, trained `CountVectorizer` + `MultinomialNB` on an 80/20 random split, and reported 0.99 accuracy and spam F1-score 0.95. It also explored vocabulary vectors and tested example SMS/email messages.

This repository includes a cleaned equivalent exploratory notebook, with original notebook outputs cleared (some were large), path made relative, private-looking example messages replaced by synthetic examples, and a reproducible seed added. The refactored training script also adds stratification and serializes a single pipeline. It is therefore *not an identical execution* of the earlier notebook.
