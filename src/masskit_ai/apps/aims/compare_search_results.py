#!/usr/bin/env python
from pathlib import Path
import hydra
import logging
from omegaconf import DictConfig
from masskit.utils.hitlist import CompareRecallDCG, Hitlist
from masskit.utils.general import MassKitSearchPathPlugin
from hydra.core.plugins import Plugins


Plugins.instance().register(MassKitSearchPathPlugin)


"""
compare search results for two searches
"""


@hydra.main(config_path="conf", config_name="config_compare", version_base=None)
def compare_search_results_app(config: DictConfig) -> None:

    compare_hitlist = Hitlist().load(Path(config.input.comparison.file).expanduser())
    ground_truth_hitlist = Hitlist().load(
        Path(config.input.ground_truth.file).expanduser())

    comparison_operator = CompareRecallDCG(comparison_score=config.comparison.comparison.score.column_name,
                                           truth_score=config.comparison.ground_truth.score.column_name)
    comparison = comparison_operator(compare_hitlist, ground_truth_hitlist)

    if config.output.pkl:
        comparison.to_pickle(Path(config.output.pkl).expanduser())
    if config.output.csv:
        comparison.to_csv(Path(config.output.csv).expanduser())

    # filter out hits to queries that hit noise
    comparison = comparison[comparison['truth_max_score']
                            >= config.comparison.ground_truth.score.threshold]

    for k in comparison_operator.recall_values:
        for column in ['truth_dcg', 'comparison_dcg']:
            logging.info(
                f"dcg for {column} with hitlist length {k}:\n{comparison[(column, k)].describe()}\n")
        logging.info(
            f"recall for hitlist length {k}:\n{comparison[('recall', k)].value_counts()}\n\n")

    logging.info(f"recall 0 scores:\n"
                 f"{comparison[comparison[('recall', 1)] == 0][['truth_max_score', 'comparison_max_score']]}")

# hitlist_length = [x for x in range(1,11)]
# avg_recall = [ np.mean(a[('recall', x)].values) for x in range(1,11)]
# avg_comparison_DCG = [ np.mean(a[('comparison_dcg', x)].values) for x in range(1,11)]
# avg_truth_DCG = [ np.mean(a[('truth_dcg', x)].values) for x in range(1,11)]
# num_nonzero_recall = [ np.count_nonzero(a[('recall', x)].values) for x in range(1,11)]


if __name__ == "__main__":
    compare_search_results_app()
