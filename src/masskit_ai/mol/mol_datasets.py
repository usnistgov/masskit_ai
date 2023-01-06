from abc import abstractmethod
from masskit.utils.arrow import save_to_arrow
from masskit_ai.base_datasets import BaseDataset
from masskit_ai.lightning import get_pytorch_ranks


class MolPropDataset(BaseDataset):
    """
    class for accessing a dataframe of small molecule and properties

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

    def get_column(self, column):
        """
        retrieve a column from the parquet file

        :param column: the column to retrieve
        """
        raise NotImplementedError()
        
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

    def get_y(self, data_row):
        """
        given the data row, return the target of the network
        """
        return data_row[self.config.output_column]
