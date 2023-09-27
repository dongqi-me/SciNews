import spacy
import json
import nltk
import multiprocessing as mp




def calc_unique_token_stats(input_data):
    # avg sentence number & word number
    if not input_data:
        raise ValueError("Input data cannot be empty.")

    vocab_article = []
    vocab_reference = []



    references = [pair['reference'].strip() for pair in input_data]
    articles = [pair['article'].strip() for pair in input_data]

    with mp.Pool(4) as pool:
        vocab_article = pool.map(nltk.tokenize.word_tokenize, articles)
        vocab_reference = pool.map(nltk.tokenize.word_tokenize, references)

    unique_token_article = [word for doc in vocab_article for word in list(set(doc))]  
    unique_token_reference = [word for doc in vocab_reference for word in list(set(doc))]

    doc_num = len(input_data)
    #unique_token_article = list(set(unique_token_reference))
    #unique_token_reference = list(set(unique_token_reference))

    return {"article_unique_total_num":len(unique_token_article), "reference_unique_total_num":len(unique_token_reference),
            "article_unique_avg_num":round(len(unique_token_article)/doc_num, 4), "reference_unique_avg_num":round(len(unique_token_reference)/doc_num, 4)}


if __name__ == "__main__":
    with open("train.json") as f:
        train_data=json.load(f)

    train_data = [{"article":instance['Paper_Body'], "reference": instance['News_Body']} for instance in train_data]

    #train_length_stat = calc_unique_token_stats(train_data)
    #print(train_length_stat["article_unique_total_num"], train_length_stat["reference_unique_total_num"], train_length_stat["article_unique_avg_num"], train_length_stat["reference_unique_avg_num"])
    # 2327240 323224 65.497 9.0967
    # 55079876 10869771 1550.1485 305.915
    with open("val.json") as f:
        val_data=json.load(f)

    val_data = [{"article":instance['Paper_Body'], "reference": instance['News_Body']} for instance in val_data]

    #val_length_stat = calc_unique_token_stats(val_data)
    #print(val_length_stat["article_unique_total_num"], val_length_stat["reference_unique_total_num"], val_length_stat["article_unique_avg_num"], val_length_stat["reference_unique_avg_num"])
    # 532075 87159 119.8367 19.6304
    # 6874042 1353107 1548.2077 304.7538

    with open("test.json") as f:
        test_data=json.load(f)

    test_data = [{"article":instance['Paper_Body'], "reference": instance['News_Body']} for instance in test_data]

    #test_length_stat = calc_unique_token_stats(test_data)
    #print(test_length_stat["article_unique_total_num"], test_length_stat["reference_unique_total_num"], test_length_stat["article_unique_avg_num"], test_length_stat["reference_unique_avg_num"])
    # 532313 88618 119.7016 19.9276
    # 6859832 1363254 1542.5752 306.5559

    with open("statistics_results/unique_token.json", 'w') as f:
        data = train_data+val_data+test_data
        data_unique_stat = calc_unique_token_stats(data)
        json.dump(data_unique_stat, f)