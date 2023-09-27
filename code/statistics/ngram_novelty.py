
import nltk
import numpy as np
import multiprocessing as mp
import json
import functools

def novel_ngram_proportion(n: int, pair):
    source, target = pair
    # First, split the target and source strings into lists of n-grams
    source_words = nltk.tokenize.word_tokenize(source)
    target_words = nltk.tokenize.word_tokenize(target)
    target_ngrams = [target_words[i:i+n] for i in range(len(target_words) - n + 1)]
    source_ngrams = [source_words[i:i+n] for i in range(len(source_words) - n + 1)]

    # Then, calculate the number of novel n-grams in the target string
    # by checking for n-grams that are in the target but not in the source
    num_novel_ngrams = sum(1 for ngram in target_ngrams if ngram not in source_ngrams)

    # Finally, calculate the proportion of novel n-grams by dividing the number
    # of novel n-grams by the total number of n-grams in the target string
    if len(target_ngrams) == 0:
        return 0.0
    return num_novel_ngrams / len(target_ngrams)

def calc_ngram_novelty(input_data):
    if not input_data:
        raise ValueError("Input data cannot be empty.")
    references = [pair['reference'].strip() for pair in input_data]
    articles = [pair['article'].strip() for pair in input_data]
    one_gram_novelty = []
    two_gram_novelty = []
    three_gram_novelty = []
    four_gram_novelty = []
    with mp.Pool(4) as pool:
        one_gram_novelty = pool.map(functools.partial(novel_ngram_proportion, 1), zip(references, articles))
        two_gram_novelty = pool.map(functools.partial(novel_ngram_proportion, 2), zip(references, articles))
        three_gram_novelty = pool.map(functools.partial(novel_ngram_proportion, 3), zip(references, articles))
        four_gram_novelty = pool.map(functools.partial(novel_ngram_proportion, 4), zip(references, articles))
    

        
    return {"1gram_novelty":round(np.mean(one_gram_novelty), 4), "2gram_novelty":round(np.mean(two_gram_novelty), 4), "3gram_novelty":round(np.mean(three_gram_novelty), 4), "4gram_novelty":round(np.mean(four_gram_novelty), 4), "1gram_list":one_gram_novelty, "2gram_list":two_gram_novelty, "3gram_list":three_gram_novelty, "4gram_list":four_gram_novelty}

if __name__ == "__main__":
    with open("train.json") as f:
        train_data=json.load(f)

    train_data = [{"article":instance['Paper_Body'], "reference": instance['News_Body']} for instance in train_data]

    #train_length_stat = calc_ngram_novelty(train_data)
    #print(train_length_stat["1gram_novelty"], train_length_stat["2gram_novelty"], train_length_stat["3gram_novelty"], train_length_stat["4gram_novelty"])
    # 0.5232 0.9128 0.9845 0.9937

    with open("val.json") as f:
        val_data=json.load(f)

    val_data = [{"article":instance['Paper_Body'], "reference": instance['News_Body']} for instance in val_data]

    #val_length_stat = calc_ngram_novelty(val_data)
    #print(val_length_stat["1gram_novelty"], val_length_stat["2gram_novelty"], val_length_stat["3gram_novelty"], val_length_stat["4gram_novelty"])
    # 0.5236 0.9133 0.9845 0.9936

    with open("test.json") as f:
        test_data=json.load(f)

    test_data = [{"article":instance['Paper_Body'], "reference": instance['News_Body']} for instance in test_data]

    #test_length_stat = calc_ngram_novelty(test_data)
    #print(test_length_stat["1gram_novelty"], test_length_stat["2gram_novelty"], test_length_stat["3gram_novelty"], test_length_stat["4gram_novelty"])
    # 0.5225 0.9125 0.9845 0.9936


    with open("statistics_results/ngram_novelty.json", 'w') as f:
        data = train_data+val_data+test_data
        data_ngram_stat = calc_ngram_novelty(data)
        json.dump(data_ngram_stat, f)