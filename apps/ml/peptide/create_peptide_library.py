import hydra
from masskit.peptide.spectrum_generator import generate_peptide_library
from masskit.utils.files import spectra_to_array
import pyarrow.parquet as pq
from masskit_ai.loggers import filter_pytorch_lightning_warnings
from masskit.data_specs.spectral_library import LibraryAccessor

"""
command line for creating a theoretical peptide spectra library for pretraining
"""


@hydra.main(config_path="conf", config_name="config_predict")
def main(config):

    df = generate_peptide_library(num=config.num, min_length=config.min_length, max_length=config.max_length,
                                  min_charge=config.min_charge, max_charge=config.max_charge,
                                  min_ev=config.min_ev, max_ev=config.max_ev, mod_list=config.mod_list,
                                  set='train')

    if "csv" in config.output_suffixes:
        df.to_csv(f"{config.output_prefix}.csv")
    if "pkl" in config.output_suffixes:
        df.to_pickle(f"{config.output_prefix}.pkl")
    if "parquet" in config.output_suffixes:
        table = spectra_to_array(df['spectrum'])
        pq.write_table(table, f"{config.output_prefix}.parquet", row_group_size=5000)
    if "msp" in config.output_suffixes:
        df.lib.to_msp(f"{config.output_prefix}.msp", spectrum_column='spectrum', annotate=True)

if __name__ == "__main__":
    filter_pytorch_lightning_warnings()
    main()
