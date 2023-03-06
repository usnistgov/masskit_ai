import pyarrow.parquet as pq
import numpy as np
import torch
from masskit_ai.base_datasets import BaseDataset, DataframeDataset
from masskit.data_specs.spectral_library import *
from masskit_ai.lightning import get_pytorch_ranks
from masskit.utils.arrow import save_to_arrow

"""
pytorch datasets for spectra.
note on terminology: dataloaders iterate over datasets.  samplers are used by dataloaders to sample from datasets.
datamodules set up and use dataloaders, samplers, and datasets.
"""


class SpectrumDataset(BaseDataset):
    """
    Base spectrum dataset
    """
    def __init__(self, store_in, config_in, set_to_load, output_column=None, columns=None) -> None:
        """
        :param store_in: data store
        :param config_in: configuration data
        :param set_to_load: which set to load, e.g. train, valid, test
        :param output_column: the name of the column to use for output
        :param columns: columns to load.  otherwise, use ms.columns
        """
        super().__init__(store_in, config_in, set_to_load, output_column=output_column, columns=columns)

    def get_y(self, data_row):
        shape = (1, int(self.config.ms.max_mz / self.config.ms.bin_size))
        spectra = np.zeros(shape, dtype=np.float32)
        query = data_row[self.output_column]
        query.products.ions2array(
            spectra,
            0,
            bin_size=self.config.ms.bin_size,
            down_shift=self.config.ms.down_shift,
            intensity_norm=np.max(query.products.intensity),
            channel_first=self.config.ml.embedding.channel_first,
            take_max=self.config.ms.take_max,
            take_sqrt=self.config.ms.get('take_sqrt', False),
        )
        # spectra = np.squeeze(spectra)
        return torch.from_numpy(np.asarray(spectra))


class TandemDataframeDataset(SpectrumDataset, DataframeDataset):
    
    def __init__(self, store_in, config_in, set_to_load, output_column=None, columns=None) -> None:
        """
        :param store_in: data store
        :param config_in: configuration data
        :param set_to_load: which set to load, e.g. train, valid, test
        :param columns: columns to load.  otherwise, use ms.columns
        :param output_column: the name of the column to use for output
        """
        SpectrumDataset.__init__(self, store_in, config_in, set_to_load, output_column=output_column, columns=columns)
        DataframeDataset.__init__(self, store_in, config_in, set_to_load, output_column=output_column, columns=columns)


class TandemArrowDataset(SpectrumDataset):
    """
    class for accessing a tandem dataframe of spectra

    How workers are set up requires some explanation:

    - if there is more than one gpu, each gpu has a corresponding main process.
    - the numbering of this gpu within a node is given by the environment variable LOCAL_RANK
    - if there is more than one node, the numbering of the node is given by the NODE_RANK environment variable
    - the number of nodes times the number of gpus is given by the WORLD_SIZE environment variable
    - the number of gpus on the current node can be found by parsing the PL_TRAINER_GPUS environment variable
    - these environment variables are only available when doing ddp.  Otherwise sharding should be done using id and num_workers in torch.utils.data.get_worker_info()
    - each main process creates an instance of Dataset.  This instance is NOT initialized by worker_init_fn, only the constructor.
    - each worker is created by forking the main process, giving each worker a copy of Dataset, already constructed.
      - each of these forked Datasets is initialized by worker_init_fn
      - the global torch.utils.data.get_worker_info() contains a reference to the forked Dataset and other info
      - these workers then take turns feeding minibatches into the training process
      - ***important*** since each worker is a copy, __init__() is only called once, only in the main process
    - the dataset in the main processes is used by other steps, such as the validation callback
      - this means that if there is any important initialization done in worker_init_fn, it must explicitly be done to the main process Dataset
    - alternative sources of parameters:
       - global_rank = trainer.node_rank * trainer.nudatasetm_processes + process_idx
       - world_size = trainer.num_nodes * trainer.num_processes
    """

    def __init__(self, store_in, config_in, set_to_load, columns=None) -> None:
        """
        :param store_in: data store
        :param config_in: configuration data
        :param set_to_load: which set to load, e.g. train, valid, test
        :param columns: columns to load.  otherwise, use ms.columns
        """
        super().__init__(store_in, config_in, set_to_load, columns=columns)
        
        # set up where clause
        
        if self.config.input[self.set_to_load].where is not None:
            self.filters = eval(self.config.input[self.set_to_load].where)
        else:
            self.filters = None

        # if multiple nodes and gpus, slice the data with equal slices to each gpu
        is_parallel, world_rank, world_size, num_gpus, num_nodes, node_rank, local_rank, worker_id = get_pytorch_ranks()
        if is_parallel:
            raise NotImplementedError('distributed training not yet implemented')
        #    where += f" AND ROWID % {world_size} = {world_rank}"

    def get_column(self, column):
        """
        retrieve a column from the parquet file

        :param column: the column to retrieve
        """
        table = pq.read_table(self.store, columns=[column], filters=self.filters)
        return table[column].to_numpy()
        
    def __len__(self) -> int:
        """
        This returns the number of rows of data for the *entire* gpu process, not necessarily the number of rows
        in self.data

        :return: number of rows
        """
        return len(self.data)

    @property
    def data(self):
        if self._data is None:
            self._data = save_to_arrow(self.store, columns=self.columns, filters=self.filters)
        return self._data
        
    def to_pandas(self):
        return self.data.to_pandas()
    
    def get_data_row(self, index):
        """
        given the index, return corresponding data for the index
        """
        return self.data.getitem_by_row(index)


# collate_fn can be used to pad out minibatches
# https://www.speechmatics.com/wp-content/uploads/2019/10/Speechmatics_Dataloader_Pytorch_Ebook_2019.pdf


"""
Notes on pyarrow usage:
- I believe the sampler automatically selects for each thread a list of rows(index) for each thread.  index2id is used
to turn this into an id for sql retrieval.
- can get individual row using row = mytable.slice(100000,1) or an id using mytable.filter(bool expression)
  - then convert row = row.to_pydict()
  - then {key: value[0] for key, value in dd.items()} to get rid of arrays
  - need to create spectrum out of arrays of mz, intensity
   - if using mol, convert json to mol.
- either we have to write a client/server where the server holds the pyarrow table or override the pytorch lightning 
machinery to distribute a set of ids or rows to each thread.  each thread then loads in the entire pyarrow table, 
but then quickly filters it down to the subset that is needed.
- filtering by row at load time is only supported through the experimental pa.dataset api.  In the lightning Dataset
  code, we'd first load in the column of ids from the parquet file, then create a filter based on the list
  of ids given to the Dataset.  Then we'd use the filter using the pa.dataset filter functionality to read in a subset
  of the data.  get_data_row() would be modified using the above logic for getting a single row.
- note that gpu level slicing of the data is done in the constructor
- problem: it looks like lightning gives each thread a subset of record indices (not ids) at the time of batch
  creation.  This makes it not possible to subset the data beforehand as the list of ids is not know.  It may be
  necessary to subset and shuffle the data in worker_init_fn() by hand, including subsetting by gpu and worker, and 
  not using the weighted, random, or distributed sampler.  It depends on how these samplers work.
  - it appears that RandomSampler just randomizes a list and takes batches:
    https://github.com/pytorch/pytorch/blob/c371542efc31b1abfe6f388042aa3ab0cef935f2/torch/csrc/api/src/data/samplers/random.cpp#L12
    implying that the sampler is applied after the list of indices is sliced up
  - in DistributedRandomSampler, it appears that each instance gets a subset of the indices. 
    https://github.com/pytorch/pytorch/blob/e3d75b8475a1668a02ac4a23c160b7aee8ebb3d3/torch/csrc/api/src/data/samplers/distributed.cpp#L15
    https://github.com/pytorch/pytorch/blob/b2e79ed5ecabcf4be299dc2ed085223ab5c22fd7/torch/csrc/api/include/torch/data/samplers/distributed.h#L54
  - it doesn't appear that there is an easy way for the dataset to get the information in the sampler (or the
   dataloader). This can be worked around in MasskitDataModule.create_loader() where the sampler and dataset is created
   by creating a dict that matches worker_id to index start stop by examining the contents of each sampler
   private variable indices_.  This is suboptimal as this is a private value and may not even be available in python.
   dict is then passed to the sampler on construction. 
  - it's still not clear where the sampler is set up to get the right indices per worker. I don't see how it can be in
    create_loader().  Is the sampler adjusted after the fact?
"""

