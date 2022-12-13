import hydra
import logging
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from masskit.utils.hitlist import CompareRecallDCG, Hitlist

"""
compare search results for two searches
"""


@hydra.main(config_path="conf", config_name="config_compare")
def compare_search_results_app(config: DictConfig) -> None:

    compare_hitlist = Hitlist().load(config.input.comparison.file)
    ground_truth_hitlist = Hitlist().load(config.input.ground_truth.file)

    comparison_operator = CompareRecallDCG(comparison_score=config.comparison.comparison.score.column_name,
                                           truth_score=config.comparison.ground_truth.score.column_name)
    comparison = comparison_operator(compare_hitlist, ground_truth_hitlist)

    if config.output.pkl:
        comparison.to_pickle(config.output.pkl)
    if config.output.csv:
        comparison.to_csv(config.output.csv)

    # filter out hits to queries that hit noise
    comparison = comparison[comparison['truth_max_score'] >= config.comparison.ground_truth.score.threshold]

    for k in comparison_operator.recall_values:
        for column in ['truth_dcg', 'comparison_dcg']:
            logging.info(f"dcg for {column} with hitlist length {k}:\n{comparison[(column, k)].describe()}\n")
        logging.info(f"recall for hitlist length {k}:\n{comparison[('recall', k)].value_counts()}\n\n")

    logging.info(f"recall 0 scores:\n"
                 f"{comparison[comparison[('recall', 1)] == 0][['truth_max_score', 'comparison_max_score']]}")

# hitlist_length = [x for x in range(1,11)]
# avg_recall = [ np.mean(a[('recall', x)].values) for x in range(1,11)]
# avg_comparison_DCG = [ np.mean(a[('comparison_dcg', x)].values) for x in range(1,11)]
# avg_truth_DCG = [ np.mean(a[('truth_dcg', x)].values) for x in range(1,11)]
# num_nonzero_recall = [ np.count_nonzero(a[('recall', x)].values) for x in range(1,11)]

if __name__ == "__main__":
    compare_search_results_app()
