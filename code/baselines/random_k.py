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

def random_k_baseline(input_data, avg_tok_num=695):
    if not input_data:
        raise ValueError("Input data cannot be empty.")
    all_pairs = []
    for pair in tqdm(input_data):
        try:
            article = pair['article'].strip()

        except KeyError:
            raise ValueError("Each dictionary in input data should contain 'reference' and 'prediction' keys.")
        
        
        article_sents = nltk.tokenize.sent_tokenize(article)
        id_list = [i for i in range(len(article_sents))]
        random.shuffle(id_list)
        selected_ids = []
        
        cur_len = 0
        for id in id_list:
            
            tok_num = len(nltk.tokenize.word_tokenize(article_sents[id])) 
            if cur_len + tok_num <= avg_tok_num:
                cur_len += tok_num
                selected_ids.append(id)
            else:
                break
        selected_ids.sort()
        pair['prediction'] = ' '.join([article_sents[i] for i in selected_ids])
        
        all_pairs.append(pair)
    return all_pairs

if __name__ == "__main__":
    set_seed(2023)
    with open("test.json") as f:
        test_data=json.load(f)

    test_data = [{"article":instance['Paper_Body'], "reference": instance['News_Body']} for instance in test_data]
    test_output = random_k_baseline(test_data)

    with open("random_k_output.json", 'w') as f:
        json.dump(test_output, f)