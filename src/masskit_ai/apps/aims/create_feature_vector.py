#!/usr/bin/env python
from pathlib import Path
import hydra
from hydra.utils import to_absolute_path
import pyarrow as pa
import pyarrow.parquet as pq
from masskit.utils.tablemap import ArrowLibraryMap
from masskit_ai.base_objects import ModelInput
from masskit_ai.spectrum.small_mol.small_mol_lightning import SearchLightningModule
import numpy as np
import torch
from masskit.utils.general import MassKitSearchPathPlugin, read_arrow, write_arrow
from hydra.core.plugins import Plugins


Plugins.instance().register(MassKitSearchPathPlugin)


"""
Use an ML model to create a feature vector column in a parquet file
"""


@hydra.main(config_path="conf", config_name="config_create", version_base=None)
def create_feature_vector_app(config):

    # Allow files relative to original execution directory
    parquet_file = Path(to_absolute_path(config.input.file)).expanduser()

    # load the library as a LibraryMap
    table = read_arrow(parquet_file)
    annotation_map = ArrowLibraryMap(table)

    # lightning module has to be configurable
    model = SearchLightningModule.load_from_checkpoint(
        Path(config.input.checkpoint).expanduser())
    model.eval()  # eval mode turns off training flag in all layers of model

    feature_vectors = []

    for k in range(len(annotation_map)):

        shape = (1, int(model.config.ms.max_mz / model.config.ms.bin_size))
        spectrum_array = np.zeros(shape, dtype=np.float32)
        spectrum = annotation_map[k]['spectrum']
        spectrum.products.ions2array(
            spectrum_array,
            0,
            bin_size=model.config.ms.bin_size,
            down_shift=model.config.ms.down_shift,
            intensity_norm=np.max(spectrum.products.intensity),
            channel_first=model.config.ml.embedding.channel_first,
            take_max=model.config.ms.take_max,
            take_sqrt=model.config.ms.get('take_sqrt', False),
        )
        spectrum_array = torch.from_numpy(spectrum_array)
        with torch.no_grad():
            x = ModelInput(x=spectrum_array, y=None, index=None)
            output = model(x)
            feature_vector = output.y_prime[0, :].detach().numpy()
            feature_vector_size = feature_vector.shape[-1]
            feature_vectors.append(feature_vector)

    if table.schema.get_field_index(config.output.column_name) == -1:
        table = table.append_column(config.output.column_name, pa.array(
            feature_vectors, pa.large_list(pa.float32())))
    else:
        i = table.schema.get_field_index(config.output.column_name)
        table = table.set_column(i, config.output.column_name, pa.array(
            feature_vectors, pa.large_list(pa.float32())))
    # add metadata
    i = table.schema.get_field_index(config.output.column_name)
    table = table.cast(table.schema.set(i, table.schema.field(i).with_metadata(
        {"fp_size": feature_vector_size.to_bytes(8, byteorder='big')})))

    write_arrow(table, Path(config.output.file).expanduser(), row_group_size=5000)


if __name__ == "__main__":
    create_feature_vector_app()
