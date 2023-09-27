
import nltk
import numpy as np
import pandas as pd
import json

import argparse
from summac.model_summac import SummaCConv


def main(args):
    model_conv = SummaCConv(models=["vitc"], bins='percentile', granularity="sentence", nli_labels="e", device="cuda", start_file="default", agg="mean")
    # print("model loaded")
    with open(args.model_output) as f:
        data = json.load(f)
    preds = [pair['prediction'] for pair in data]
    sources = [pair['article'] for pair in data]
    refs = [pair['reference'] for pair in data]

    
    
    scores = [model_conv.score([source], [prediction])['scores'][0] for source, prediction in zip(sources, preds)]
    summacconv = {}
    summacconv['summacconv_list'] = scores
    summacconv['summacconv'] = round(np.mean(scores), args.round)
    print(np.mean(scores))
    prefix = ''.join(args.model_output.split('.json')[:-1])
    result_file_name = prefix + '_summacconv.json'
    with open(result_file_name, 'w') as f:
        json.dump(summacconv, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_output', type=str, help='a csv file containing model ouputs')
    parser.add_argument('--round', type=int, default=5)
    args = parser.parse_args()
    main(args)
