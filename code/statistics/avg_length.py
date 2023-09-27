import spacy
import json
from tqdm import tqdm
import numpy as np
import nltk
import multiprocessing as mp



def calc_length_stats(input_data):
    # avg sentence number & word number
    if not input_data:
        raise ValueError("Input data cannot be empty.")


    token_article_dist = []
    token_reference_dist = []
    sent_article_dist = []
    sent_reference_dist = []

    references = [pair['reference'].strip() for pair in input_data]
    articles = [pair['article'].strip() for pair in input_data]

    with mp.Pool(16) as pool:
        token_article_dist = pool.map(nltk.tokenize.word_tokenize, articles)
        token_reference_dist = pool.map(nltk.tokenize.word_tokenize, references)

        sent_article_dist = pool.map(nltk.tokenize.sent_tokenize, articles)
        sent_reference_dist = pool.map(nltk.tokenize.sent_tokenize, references)

    token_article_dist = [len(doc) for doc in token_article_dist]
    token_reference_dist = [len(doc) for doc in token_reference_dist]
    sent_article_dist = [len(doc) for doc in sent_article_dist]
    sent_reference_dist = [len(doc) for doc in sent_reference_dist]

    
    avg_token_article = np.mean(token_article_dist)
    avg_token_reference = np.mean(token_reference_dist)
    avg_sent_article = np.mean(sent_article_dist)
    avg_sent_reference = np.mean(sent_reference_dist)

    return {"avg_token_article":round(avg_token_article, 4), "avg_token_reference":round(avg_token_reference, 4), "avg_sent_article":round(avg_sent_article, 4), "avg_sent_reference":round(avg_sent_reference, 4),
            "token_article_dist":token_article_dist, "token_reference_dist":token_reference_dist, "sent_article_dist":sent_article_dist, "sent_reference_dist":sent_reference_dist}

if __name__ == "__main__":
    with open("train.json") as f:
        train_data=json.load(f)

    train_data = [{"article":instance['Paper_Body'], "reference": instance['News_Body']} for instance in train_data]

    #train_length_stat = calc_length_stats(train_data)
    #print(train_length_stat["avg_token_article"], train_length_stat["avg_token_reference"], train_length_stat["avg_sent_article"], train_length_stat["avg_sent_reference"])
    # 7766.9098 695.0924 290.6458 25.1779

    with open("val.json") as f:
        val_data=json.load(f)

    val_data = [{"article":instance['Paper_Body'], "reference": instance['News_Body']} for instance in val_data]

    #val_length_stat = calc_length_stats(val_data)
    #print(val_length_stat["avg_token_article"], val_length_stat["avg_token_reference"], val_length_stat["avg_sent_article"], val_length_stat["avg_sent_reference"])
    # 7751.0128 691.1018 291.2491 25.0275

    with open("test.json") as f:
        test_data=json.load(f)

    test_data = [{"article":instance['Paper_Body'], "reference": instance['News_Body']} for instance in test_data]

    #test_length_stat = calc_length_stats(test_data)
    #print(test_length_stat["avg_token_article"], test_length_stat["avg_token_reference"], test_length_stat["avg_sent_article"], test_length_stat["avg_sent_reference"])
    # 7722.7207 696.1936 288.7931 25.2863

    with open("statistics_results/avg_length.json", 'w') as f:
        data = train_data+val_data+test_data
        data_len_stat = calc_length_stats(data)
        json.dump(data_len_stat, f)
    
    