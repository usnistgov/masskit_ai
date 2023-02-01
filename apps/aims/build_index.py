import hydra
import logging
from omegaconf import DictConfig, OmegaConf
from pyarrow import parquet as pq
from masskit.data_specs.schemas import min_spectrum_fields
from masskit.utils.index import ArrowLibraryMap, DescentIndex, PandasLibraryMap
from masskit.utils.general import class_for_name

"""
create index for spectral searching
"""


@hydra.main(config_path="conf", config_name="config_build", version_base=None)
def build_index_app(config: DictConfig) -> None:

    # load the library as a LibraryMap
    columns = min_spectrum_fields.copy()
    for x in [config.input.library.column_name, config.input.library.column_count_name]:
        if x is not None:
            columns.append(x)
    library_map = ArrowLibraryMap(
        pq.read_table(config.input.library.file),
        column_name=config.input.library.column_name, num=config.input.library.num)

    # initialize the index
    index = class_for_name(config.paths.modules.indices,
                           config.creation.index_type.name)(config.creation.name, dimension=config.creation.dimension)
    # create the index
    if config.creation.index_type.input_type == 'fingerprint':
        index.create_from_fingerprint(library_map, config.input.library.column_name, config.input.library.column_count_name)
    else:
        index.create(library_map)

    # save the index to the file
    index.save(file=config.output.file)


if __name__ == "__main__":
    build_index_app()
