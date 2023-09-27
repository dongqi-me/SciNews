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

def lead_three_baseline(input_data, n_sent=3):
    if not input_data:
        raise ValueError("Input data cannot be empty.")
    all_pairs = []
    for pair in tqdm(input_data, desc='Processing input data'):
        try:
            article = pair['article'].strip()
        except KeyError:
            raise ValueError("Each dictionary in input data should contain 'article' and 'reference' keys.")
        
        article_sents = tokenize.sent_tokenize(article)
        if len(article_sents) >= n_sent:
            pair['prediction'] = ' '.join(article_sents[:n_sent])
        else:
            pair['prediction'] = article
        all_pairs.append(pair)
    return all_pairs

if __name__ == "__main__":
    # Set the random seeds
    set_seed(2023)

    # Set the input and output file paths
    input_file = "test.json"
    output_file = "lead_3_output.json"

    with open(input_file) as f:
        test_data = json.load(f)

    test_data = [{"article":instance['Paper_Body'], "reference": instance['News_Body']} for instance in test_data]
    test_output = lead_three_baseline(test_data)

    with open(output_file, 'w') as f:
        json.dump(test_output, f, indent=4)


