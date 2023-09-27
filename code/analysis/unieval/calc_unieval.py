from utils import convert_to_json
from metric.evaluator import get_evaluator
import json
import argparse
import numpy as np

def main(args):
    task = 'fact'
    with open(args.model_output) as f:
        data = json.load(f)

    new_data = [{"system_output": pair['prediction'], "source": pair['article']} for pair in data]
    # Initialize evaluator for a specific task
    evaluator = get_evaluator(task)
    # Get factual consistency scores
    eval_scores = evaluator.evaluate(new_data, print_result=True)
    scores = [score['consistency'] for score in eval_scores]

    unievalscores = {}
    unievalscores['unievalscores_list'] = scores
    unievalscores['unievalscore'] = round(np.mean(scores), args.round)
    print(np.mean(scores))
    prefix = ''.join(args.model_output.split('.json')[:-1])
    result_file_name = prefix + '_unievalscores.json'
    with open(result_file_name, 'w') as f:
        json.dump(unievalscores, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_output', type=str, help='a json file containing model ouputs')
    parser.add_argument('--round', type=int, default=5)
    args = parser.parse_args()
    main(args)