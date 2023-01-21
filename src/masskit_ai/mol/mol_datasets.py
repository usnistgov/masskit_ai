from abc import abstractmethod
import torch
from masskit.utils.arrow import save_to_arrow
from masskit_ai.base_datasets import BaseDataset
from masskit_ai.lightning import get_pytorch_ranks
from masskit_ai.mol.small.models import path_utils
from masskit_ai.mol.small.models import mol_graph
from masskit_ai.base_objects import ModelInput


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
        return data_row[self.config.ml.output_column]


# collating code to make molecule data have the same shape for graphormer

def pad_1d_unsqueeze(x, padlen):
    x = x + 1  # pad id = 0
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen], dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_2d_unsqueeze(x, padlen):
    x = x + 1  # pad id = 0
    xlen, xdim = x.size()
    if xlen < padlen:
        new_x = x.new_zeros([padlen, xdim], dtype=x.dtype)
        new_x[:xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_attn_bias_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype).fill_(float("-inf"))
        new_x[:xlen, :xlen] = x
        new_x[xlen:, :xlen] = 0
        x = new_x
    return x.unsqueeze(0)


def pad_edge_type_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen, x.size(-1)], dtype=x.dtype)
        new_x[:xlen, :xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_spatial_pos_unsqueeze(x, padlen):
    x = x + 1
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype)
        new_x[:xlen, :xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_3d_unsqueeze(x, padlen1, padlen2, padlen3):
    x = x + 1
    xlen1, xlen2, xlen3, xlen4 = x.size()
    if xlen1 < padlen1 or xlen2 < padlen2 or xlen3 < padlen3:
        new_x = x.new_zeros([padlen1, padlen2, padlen3, xlen4], dtype=x.dtype)
        new_x[:xlen1, :xlen2, :xlen3, :] = x
        x = new_x
    return x.unsqueeze(0)


def graphormer_collator(config):
    """
    collation function factory for graphormer
    """
    max_node = config.ml.model.Graphormer_slim.max_nodes
    multi_hop_max_dist = config.ml.model.Graphormer_slim.multi_hop_max_dist
    spatial_pos_max = config.ml.model.Graphormer_slim.spatial_pos_max

    def inner_func(items):
        items = [item for item in items if item is not None and item.x['x'].size(0) <= max_node]
        try:
            items = [
                (
                    item.index,
                    item.x['attn_bias'],
                    item.x['attn_edge_type'],
                    item.x['spatial_pos'],
                    item.x['in_degree'],
                    item.x['out_degree'],
                    item.x['x'],
                    item.x['edge_input'][:, :, :multi_hop_max_dist, :],
                    torch.Tensor([item.y]),
                )
                for item in items
            ]
        except TypeError:
            raise
        (
            indexes,
            attn_biases,
            attn_edge_types,
            spatial_poses,
            in_degrees,
            out_degrees,
            xs,
            edge_inputs,
            ys,
        ) = zip(*items)

        for idx, _ in enumerate(attn_biases):
            attn_biases[idx][1:, 1:][spatial_poses[idx] >= spatial_pos_max] = float("-inf")
        max_node_num = max(i.size(0) for i in xs)
        max_dist = max(i.size(-2) for i in edge_inputs)
        y = torch.unsqueeze(torch.cat(ys) / 10000.0, dim=-1)
        x = torch.cat([pad_2d_unsqueeze(i, max_node_num) for i in xs])
        edge_input = torch.cat(
            [pad_3d_unsqueeze(i, max_node_num, max_node_num, max_dist) for i in edge_inputs]
        )
        attn_bias = torch.cat(
            [pad_attn_bias_unsqueeze(i, max_node_num + 1) for i in attn_biases]
        )
        attn_edge_type = torch.cat(
            [pad_edge_type_unsqueeze(i, max_node_num) for i in attn_edge_types]
        )
        spatial_pos = torch.cat(
            [pad_spatial_pos_unsqueeze(i, max_node_num) for i in spatial_poses]
        )
        in_degree = torch.cat([pad_1d_unsqueeze(i, max_node_num) for i in in_degrees])

        return dict(
            index=torch.LongTensor(indexes),
            attn_bias=attn_bias,
            attn_edge_type=attn_edge_type,
            spatial_pos=spatial_pos,
            in_degree=in_degree,
            out_degree=in_degree,  # for undirected graph
            x=x,
            edge_input=edge_input,
            y=y,
        )

    return inner_func


def patn_collator(config):
    """
    collation function factory for PATN
    """
    args = config.ml.model.PropPredictor

    def inner_func(items):
        """
        takes output from embedding and collates into batch tensors and creates molgraphs 
        """
        # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # don't put tensors on cuda as this doesn't work with forked processes
        device = torch.device("cpu")

        x, y, index = zip(*items)
        batch_mols, batch_n_atoms, batch_path_inputs, batch_path_masks  = zip(*x)

        max_atoms = max(batch_n_atoms)
        batch_path_inputs, batch_path_masks = path_utils.merge_path_inputs(
                batch_path_inputs, batch_path_masks, max_atoms, args)

        # batch_path_inputs = batch_path_inputs.to(device=device)
        # batch_path_masks = batch_path_masks.to(device=device)
        mol_graphs = mol_graph.MolGraph(batch_mols, batch_path_inputs, batch_path_masks)
        # todo: get rid of normalization hack
        y = torch.tensor(y, dtype=torch.float32, device=device)/10000.0
        index = torch.tensor(index, dtype=torch.int64, device=device)
        return ModelInput(x=mol_graphs, y=y, index=index)

    return inner_func
