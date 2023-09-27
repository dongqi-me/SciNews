import json
import os
import torch
import numpy as np
from bert_score import score

class BERTScoreCalculator:
    def __init__(self):
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

    def compute_bert_score(self, input_data):
        if not input_data:
            raise ValueError("Input data cannot be empty.")

        results = []
        for pair in input_data:
            try:
                reference = pair['reference'].strip()
                prediction = pair['prediction'].strip()
            except KeyError:
                raise ValueError("Each dictionary in input data should contain 'reference' and 'prediction' keys.")
            
            P, R, F1 = score([prediction], [reference], lang="en", model_type='bert-base-uncased', device=self.device)
            scores = {
                'precision': round(P.item() * 100, 4),
                'recall': round(R.item() * 100, 4),
                'f1': round(F1.item() * 100, 4)
            }
            results.append(scores)

        average_scores = {k: round(np.mean([res[k] for res in results]), 4) for k in results[0]}
        results.append({'corpus_level': average_scores})
        return results

    def compute_bert_score_from_file(self, input_json_path, output_json_path):
        if not os.path.isfile(input_json_path):
            raise FileNotFoundError(f"{input_json_path} does not exist.")

        with open(input_json_path, 'r') as f:
            data = json.load(f)
        results = self.compute_bert_score(data)

        with open(output_json_path, 'w') as f:
            json.dump(results, f, indent=4)

if __name__ == '__main__':
    bert_score_calculator = BERTScoreCalculator()

    input_data = [
        {"reference": "The cat sat on the mat. It was happy.", "prediction": "A cat was sitting on the mat. The cat seemed pleased."},
        {"reference": "The quick brown fox jumps over the lazy dog. The dog did not react.", "prediction": "A fast brown fox jumps above the lazy dog. The dog seemed indifferent."}
    ]

    input_json_path = 'input.json'
    with open(input_json_path, 'w') as f:
        json.dump(input_data, f)

    output_json_path = 'output.json'
    bert_score_calculator.compute_bert_score_from_file(input_json_path, output_json_path)
