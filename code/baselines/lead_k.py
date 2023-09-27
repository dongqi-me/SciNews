# training 长度, 保留完整的句子

import nltk
import json
from tqdm import tqdm


def lead_k_baseline(input_data, avg_tok_num=695):
    if not input_data:
        raise ValueError("Input data cannot be empty.")
    all_pairs = []
    for pair in tqdm(input_data):
        try:
            article = pair['article'].strip()

        except KeyError:
            raise ValueError("Each dictionary in input data should contain 'reference' and 'prediction' keys.")
        
        
        sents = nltk.tokenize.sent_tokenize(article)
        lead_k_sents = []
        cur_len = 0
        for sent in sents:
            
            tok_num = len(nltk.tokenize.word_tokenize(sent)) 
            if cur_len + tok_num <= avg_tok_num:
                cur_len += tok_num
                lead_k_sents.append(sent)
            else:
                break
        
        pair['prediction'] = ' '.join(lead_k_sents)
        
        all_pairs.append(pair)
    return all_pairs

if __name__ == "__main__":

    with open("test.json") as f:
        test_data=json.load(f)

    test_data = [{"article":instance['Paper_Body'], "reference": instance['News_Body']} for instance in test_data]
    test_output = lead_k_baseline(test_data)

    with open("lead_k_output.json", 'w') as f:
        json.dump(test_output, f)