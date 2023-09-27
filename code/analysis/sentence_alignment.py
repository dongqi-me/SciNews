import evaluate
from datasets import list_datasets, load_dataset
import nltk
import numpy as np
import pandas as pd
from collections import defaultdict
import pickle
import argparse
import json

from multiprocessing import Pool

class SentenceAligner:
    def __init__(self, num_process=10, indicator='rougeL'):
        self.scorer = evaluate.load('rouge')
        self.indicator = indicator
        self.num_process = num_process

    def _align(self, article: str, summary: str):
        article_sents = nltk.sent_tokenize(article)
        summary_sents = nltk.sent_tokenize(summary)
        align_positions = []
        article_length = len(article_sents)
        for sent in summary_sents:
            best_score = 0.0
            align_pos = -1
            for i in range(article_length):
                score = self.scorer.compute(predictions=[sent], references=[article_sents[i]], rouge_types=[self.indicator])[self.indicator]
                if score > best_score:
                    best_score = score
                    align_pos = i
            align_positions.append(align_pos+1)
        return {"article_length": article_length, "align_positions": align_positions}
    
    def align(self, articles: list, summaries: list):
        with Pool(processes=self.num_process) as pool:
            results = pool.starmap(self._align, zip(articles, summaries))
        sum_summary_sentences = 0
        ratio = {}
        for i in range(10):
            ratio[str(i)] = 0
        for result in results:
            
            sum_summary_sentences += len(result['align_positions'])
            for pos in result['align_positions']:
                ratio[str(int(10 * pos/result['article_length']))] += 1
        for k, v in ratio.items():
            ratio[k] = v / sum_summary_sentences
        return ratio


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_output', type=str, help='a json file containing model ouputs')
    parser.add_argument('--num_workers', type=int, default=30)
    args = parser.parse_args()

    with open(args.model_output) as f:
        data = json.load(f)
    preds = [pair['prediction'] for pair in data]
    sources = [pair['article'] for pair in data]
    refs = [pair['reference'] for pair in data]
    
    SA = SentenceAligner(args.num_workers)
    ratio = SA.align(sources, preds)
    print(ratio)
    prefix = prefix = ''.join(args.model_output.split('.json')[:-1])
    result_file_name = prefix + '_alignment_distribution.json'
    with open(result_file_name, 'w') as f:
        json.dump(ratio, f)
   