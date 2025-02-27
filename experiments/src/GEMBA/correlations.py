import json
from collections import defaultdict
import scipy.stats
import numpy as np
import argparse
import os
import glob

def calculate_correlations(results_dir):

    # get all files in results_dir
    for model in os.listdir(results_dir):
        print(model)
        for lp in os.listdir(os.path.join(results_dir, model)):
            print(lp)
            
            # store results for each lp, with prompt_id as key, with a dict of scores + p-values
            lp_results = defaultdict(dict)

            # get all files in the second level
            for file in glob.glob(os.path.join(results_dir, model, lp, '*.jsonl')):

                # Store scores by system
                system_human_scores = defaultdict(list)
                system_prompt_scores = defaultdict(list)
                
                # Store all segment scores
                all_human_scores = []
                all_prompt_scores = []

                number_of_retries = 0
                number_unsuccessful_retries = 0

                with open(file) as f:
                    for line in f:
                        data = json.loads(line)
                        scores = data['scores']
                        prompt_id = data['prompt_id']
                        
                        # Extract scores for each system
                        for system, system_scores in scores.items():
                            human_score = system_scores['human']
                            prompt_score = system_scores[prompt_id]
                            
                            system_human_scores[system].append(human_score)
                            system_prompt_scores[system].append(prompt_score[0] if prompt_score[0] else 0)
                            number_of_retries += prompt_score[1]

                            if prompt_score[0] is None:
                                number_unsuccessful_retries += 1
                            
                            all_human_scores.append(human_score)
                            # score[0] is the score, score[1] is the answer_id
                            # if no score, append 0
                            all_prompt_scores.append(prompt_score[0] if prompt_score[0] else 0)
                
                # Calculate segment-level correlation
                segment_correlation, segment_p_value = scipy.stats.pearsonr(all_human_scores, all_prompt_scores)
                
                # Calculate system-level correlation
                system_human_avgs = []
                system_prompt_avgs = []
                
                for system in system_human_scores:
                    system_human_avgs.append(np.mean(system_human_scores[system]))
                    system_prompt_avgs.append(np.mean(system_prompt_scores[system]))
                
                system_correlation, system_p_value = scipy.stats.pearsonr(system_human_avgs, system_prompt_avgs)
                
                average_retries = number_of_retries / len(all_prompt_scores)
                average_unsuccessful_retries = number_unsuccessful_retries / len(all_prompt_scores)

                lp_results[prompt_id] = {
                    'segment_correlation': segment_correlation,
                    'segment_p_value': segment_p_value,
                    'system_correlation': system_correlation,
                    'system_p_value': system_p_value,
                    'average_retries': average_retries,
                    'average_unsuccessful_retries': average_unsuccessful_retries,
                    'raw_retries': number_of_retries,
                    'raw_unsuccessful_retries': number_unsuccessful_retries
                }
        
            # save results for each lp, model
            output_path = os.path.join(results_dir, model, lp, 'correlations.json')
            print(output_path)
            with open(output_path, 'w') as f:
                json.dump(lp_results, f)


def plot_results(results_dir):
    
    raise NotImplementedError("Not implemented")



def main():
    # Read command line arguments
    parser = argparse.ArgumentParser(description='Calculate correlations between human and model scores')
    parser.add_argument('--results_dir', default="scores/wmt24", type=str, required=False, help='Path to the folder containing scores')
    args = parser.parse_args()

    # Calculate correlations, save results for each lp, model
    correlations = calculate_correlations(args.results_dir)

    # plot_results(args.results_dir)

if __name__ == "__main__":
    main()

# usage: python -m correlations --results_dir scores/wmt24