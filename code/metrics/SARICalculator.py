import json
import os
import numpy as np
import evaluate

class SARICalculator:
    def __init__(self):
        self.sari = evaluate.load('sari')

    def compute_sari(self, input_data):
        if not input_data:
            raise ValueError("Input data cannot be empty.")

        results = []
        for pair in input_data:
            try:
                source=pair['source'].strip()
                reference = pair['reference'].strip()
                prediction = pair['prediction'].strip()
            except KeyError:
                raise ValueError("Each dictionary in input data should contain 'source', 'reference' and 'prediction' keys.")
                
            scores = self.sari.compute(sources=[source], predictions=[prediction], references=[[reference]])

            # Round the results to 4 decimal places
            scores = {k: round(v , 4) for k, v in scores.items()}
            results.append(scores)

        average_scores = {k: round(np.mean([res[k] for res in results]), 4) for k in results[0]}
        results.append({'corpus_level': average_scores})
        return results

    def compute_sari_from_file(self, input_json_path, output_json_path):
        if not os.path.isfile(input_json_path):
            raise FileNotFoundError(f"{input_json_path} does not exist.")

        with open(input_json_path, 'r') as f:
            data = json.load(f)
        results = self.compute_sari(data)
        print(results)
        with open(output_json_path, 'w') as f:
            json.dump(results, f, indent=4)

if __name__ == '__main__':

    sari_calculator = SARICalculator()

    input_data = [
        {"source":"The quick brown fox jumps over the lazy dog. The dog did not react.", "reference": "The cat sat on the mat. It was happy.", "prediction": "A cat was sitting on the mat. The cat seemed pleased."},
    ]

    input_json_path = 'input.json'
    with open(input_json_path, 'w') as f:
        json.dump(input_data, f)

    output_json_path = 'output.json'
    sari_calculator.compute_sari_from_file(input_json_path, output_json_path)
