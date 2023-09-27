import nltk
import json
from tqdm import tqdm
import random
import numpy as np
import os
# Set random seeds for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)

def random_three_baseline(input_data, n_sent=3):
    if not input_data:
        raise ValueError("Input data cannot be empty.")
    all_pairs = []
    for pair in tqdm(input_data):
        try:
            article = pair['article'].strip()

        except KeyError:
            raise ValueError("Each dictionary in input data should contain 'reference' and 'prediction' keys.")
        
        article_sents = nltk.tokenize.sent_tokenize(article)
        if len(article_sents) >= n_sent:
            id_list = [i for i in range(len(article_sents))]
            sampled_id = random.sample(id_list, n_sent)
            sampled_id.sort()
        
            pair['prediction'] = ' '.join([article_sents[i] for i in sampled_id])
        else:
            pair['prediction'] = article
        all_pairs.append(pair)
    return all_pairs

if __name__ == "__main__":
    set_seed(2023)
    with open("test.json") as f:
        test_data=json.load(f)

    test_data = [{"article":instance['Paper_Body'], "reference": instance['News_Body']} for instance in test_data]
    test_output = random_three_baseline(test_data)

    with open("random_3_output.json", 'w') as f:
        json.dump(test_output, f)