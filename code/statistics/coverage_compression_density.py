import numpy as np
from multiprocessing import Pool
import nltk




class SummarizationCharacterScorer:
    """
    This is an implementation of three summarization characters: coverage, density and compression ratio which are intented to measure the extractiveness of summaries
    Ideas from Grusky et al. 2018
    """
    def __init__(self, num_process=5):
        self.num_process = num_process


    def _compute(self, article: str, summary:str)-> tuple:


        article_words = nltk.tokenize.word_tokenize(article)
        summary_words = nltk.tokenize.word_tokenize(summary)

        if len(summary_words) == 0:  # in case some summaries is empty
            return 0.0, 0.0, 0.0
        shared_sequence = self._get_extractive_segments(article, summary)
        coverage = np.sum([len(nltk.tokenize.word_tokenize(sequence)) for sequence in shared_sequence]) / len(summary_words)
        density = np.sum([len(nltk.tokenize.word_tokenize(sequence)) ** 2 for sequence in shared_sequence]) / len(summary_words)
        compression_ratio = len(article_words) / len(summary_words)
        # print(coverage, density, compression_ratio, flush=True)
        return coverage, density, compression_ratio

    def _get_extractive_segments(self, article: str, summary: str)-> list:
        """
        greedily identify these extractive fragments of an article-summary pair
        algorithm taken from Grusky et al. 2018
        """
        article_words = nltk.tokenize.word_tokenize(article)
        summary_words = nltk.tokenize.word_tokenize(summary)
        shared_sequences = []
        i = 0
        j = 0
        while i < len(summary_words):
            shared_tokens = []
            
            while j < len(article_words):
                if summary_words[i] == article_words[j]:
                    i_end, j_end = i, j
                    while i_end < len(summary_words) and j_end < len(article_words) and summary_words[i_end] == article_words[j_end]:
                       
                        i_end, j_end = i_end + 1, j_end + 1
                   
                    if len(shared_tokens) < i_end - i:
                        shared_tokens = summary_words[i:i_end]
                    j = j_end
                else:
                    j = j + 1
           
            i, j = i + np.max([len(shared_tokens), 1]), 0
            if len(shared_tokens) > 0:
                shared_sequences.append(' '.join(shared_tokens))
       
        return shared_sequences

    def compute(self, articles: list, summaries:list) -> dict:
        coverage_list = []
        density_list = []
        compression_ratio_list = []
        with Pool(processes=self.num_process) as pool:

            res = pool.starmap(self._compute, zip(articles, summaries))
        for coverage, density, compression_ratio in res:
            
            coverage_list.append(coverage)
            density_list.append(density)
            compression_ratio_list.append(compression_ratio)
        return {"coverage":round(np.mean(coverage_list), 4), "density":round(np.mean(density), 4), "compression_ratio":round(np.mean(compression_ratio_list), 4), "coverage_list":coverage_list, "density_list":density_list, "compression_ratio_list":compression_ratio_list}
    
def calc_fragment_stat(input_data):
    if not input_data:
        raise ValueError("Input data cannot be empty.")
    SC = SummarizationCharacterScorer(4)
    articles = [pair['article'].strip() for pair in input_data]
    summaries = [pair['reference'].strip() for pair in input_data]
    return SC.compute(articles, summaries)

        

if __name__ == "__main__":
    import json
    
    with open("train.json") as f:
        train_data=json.load(f)

    train_data = [{"article":instance['Paper_Body'], "reference": instance['News_Body']} for instance in train_data]

    #train_length_stat = calc_fragment_stat(train_data)
    #print(train_length_stat["coverage"], train_length_stat["density"], train_length_stat["compression_ratio"])
    # 0.7366 1.2493 12.7145

    with open("val.json") as f:
        val_data=json.load(f)

    val_data = [{"article":instance['Paper_Body'], "reference": instance['News_Body']} for instance in val_data]

    #val_length_stat = calc_fragment_stat(val_data)
    #print(val_length_stat["coverage"], val_length_stat["density"], val_length_stat["compression_ratio"])
    # 0.736 2.6921 12.7666

    with open("test.json") as f:
        test_data=json.load(f)

    test_data = [{"article":instance['Paper_Body'], "reference": instance['News_Body']} for instance in test_data]

    #test_length_stat = calc_fragment_stat(test_data)
    #print(test_length_stat["coverage"], test_length_stat["density"], test_length_stat["compression_ratio"])
    # 0.7347 0.9378 12.6006

    with open("statistics_results/coverage_compression_density.json", 'w') as f:
        data = train_data+val_data+test_data
        data_fragment_stat = calc_fragment_stat(data)
        json.dump(data_fragment_stat, f)