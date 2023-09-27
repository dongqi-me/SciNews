# pip install summa

from summa import summarizer
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

def textrank_baseline(input_data, num_words=200):
    if not input_data:
        raise ValueError("Input data cannot be empty.")
    all_pairs = []
    for pair in tqdm(input_data, desc='Processing input data'):
        try:
            article = pair['article'].strip()
        except KeyError:
            raise ValueError("Each dictionary in input data should contain 'article' and 'reference' keys.")
        
        pair['prediction'] = summarizer.summarize(article, words=num_words).strip()
        all_pairs.append(pair)
    return all_pairs

if __name__ == "__main__":
    # Set the random seeds
    set_seed(2023)

    # Set the input and output file paths
    input_file = "test.json"
    output_file = "textrank_output.json"

    with open(input_file) as f:
        test_data = json.load(f)

    test_data = [{"article":instance['Paper_Body'], "reference": instance['News_Body']} for instance in test_data]
    test_output = textrank_baseline(test_data, num_words=695)

    with open(output_file, 'w') as f:
        json.dump(test_output, f, indent=4)