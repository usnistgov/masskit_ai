import hydra
from masskit.peptide.spectrum_generator import generate_peptide_library
from masskit.utils.files import spectra_to_array
import pyarrow.parquet as pq
from masskit_ai.loggers import filter_pytorch_lightning_warnings
from masskit.data_specs.spectral_library import LibraryAccessor

"""
command line for creating a theoretical peptide spectra library for pretraining
"""


@hydra.main(config_path="conf", config_name="config_predict", version_base=None)
def main(config):

    df = generate_peptide_library(num=config.predict.num, min_length=config.predict.min_length, max_length=config.predict.max_length,
                                  min_charge=config.predict.min_charge, max_charge=config.predict.max_charge,
                                  min_ev=config.predict.min_ev, max_ev=config.predict.max_ev, mod_list=config.predict.mod_list,
                                  set='train')

    if "csv" in config.predict.output_suffixes:
        df.to_csv(f"{config.predict.output_prefix}.csv")
    if "pkl" in config.predict.output_suffixes:
        df.to_pickle(f"{config.predict.output_prefix}.pkl")
    if "parquet" in config.predict.output_suffixes:
        table = spectra_to_array(df['spectrum'])
        pq.write_table(table, f"{config.predict.output_prefix}.parquet", row_group_size=5000)
    if "msp" in config.predict.output_suffixes:
        df['spectrum'].array.to_msp(f"{config.predict.output_prefix}.msp", annotate_peptide=True)

if __name__ == "__main__":
    filter_pytorch_lightning_warnings()
    main()
