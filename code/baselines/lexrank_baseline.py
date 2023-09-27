# pip install lexrank
from lexrank import LexRank

from lexrank.mappings.stopwords import STOPWORDS
import json
import os
import random
from nltk import tokenize
import numpy as np
from tqdm import tqdm

# Set random seeds for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)

def lexrank_baseline(input_data, lxr, n_sents=3):
    if not input_data:
        raise ValueError("Input data cannot be empty.")
    all_pairs = []
    for pair in tqdm(input_data, desc='Processing input data'):
        try:
            article = pair['article'].strip()
        except KeyError:
            raise ValueError("Each dictionary in input data should contain 'article' and 'reference' keys.")
        
        article_sents = tokenize.sent_tokenize(article)
        summary = lxr.get_summary(article_sents, summary_size=n_sents, threshold=.1)
        pair['prediction'] = ' '.join(summary)
        all_pairs.append(pair)
    return all_pairs

if __name__ == "__main__":
    # Set the random seeds
    set_seed(2023)

    # Set the input and output file paths
    input_file = "test.json"
    output_file = "lexrank_output.json"

    # learn tf-idf on training data
    train_file = "train.json"
    with open(train_file) as f:
        train_data = json.load(f)
    documents = [instance['Paper_Body'] for instance in train_data]

    lxr = LexRank(documents, stopwords=STOPWORDS['en'])

    with open(input_file) as f:
        test_data = json.load(f)

    test_data = [{"article":instance['Paper_Body'], "reference": instance['News_Body']} for instance in test_data]
    test_output = lexrank_baseline(test_data, lxr, n_sents=25) # set sentence number

    with open(output_file, 'w') as f:
        json.dump(test_output, f, indent=4)