from pathlib import Path
import hydra
from omegaconf import DictConfig
from pyarrow import parquet as pq
from masskit.utils.tablemap import ArrowLibraryMap
from masskit.utils.general import class_for_name

"""
create index for spectral searching
"""


@hydra.main(config_path="conf", config_name="config_build", version_base=None)
def build_index_app(config: DictConfig) -> None:

    # load the library as a LibraryMap
    library_map = ArrowLibraryMap(
        pq.read_table(Path(config.input.library.file).expanduser()), num=config.input.library.num)

    # initialize the index
    index = class_for_name(config.paths.modules.indices,
                           config.creation.index_type.name)(config.creation.name, dimension=config.creation.dimension)
    # create the index
    if config.creation.index_type.input_type == 'fingerprint':
        index.create_from_fingerprint(library_map, config.input.library.column_name, config.input.library.column_count_name)
    else:
        index.create(library_map, column_name=config.input.library.column_name)

    # save the index to the file
    index.save(Path(file=config.output.file).expanduser())


if __name__ == "__main__":
    build_index_app()
