from alignscore import AlignScore
import json
import argparse
import numpy as np


def main(args):
    scorer = AlignScore(model='roberta-large', batch_size=32, device='cuda:0', ckpt_path='AlignScore-large.ckpt', evaluation_mode='nli_sp') # nli_sp
    # print("model loaded")
    with open(args.model_output) as f:
        data = json.load(f)

    preds = [pair['prediction'] for pair in data]
    sources = [pair['article'] for pair in data]
    refs = [pair['reference'] for pair in data]

    
    
    scores = scorer.score(contexts=sources, claims=preds)
    
    
    alignscores = {}
    alignscores['alignscores_list'] = scores
    alignscores['alignscore'] = round(np.mean(scores), args.round)
    print(np.mean(scores))
    prefix = ''.join(args.model_output.split('.json')[:-1])
    result_file_name = prefix + '_alignscores.json'
    with open(result_file_name, 'w') as f:
        json.dump(alignscores, f)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_output', type=str, help='a csv file containing model ouputs')
    parser.add_argument('--round', type=int, default=5)
    args = parser.parse_args()
    main(args)