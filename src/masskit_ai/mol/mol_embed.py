import numpy as np
import torch
import logging
from masskit_ai.embed import Embed
from masskit_ai.mol.small.models import algos
from masskit_ai.mol.small.models import path_utils


x_map = {
    'atomic_num':
    list(range(0, 119)),
    'chirality': [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER',
        'CHI_TETRAHEDRAL',
        'CHI_ALLENE',
        'CHI_SQUAREPLANAR',
        'CHI_TRIGONALBIPYRAMIDAL',
        'CHI_OCTAHEDRAL',
    ],
    'degree':
    list(range(0, 11)),
    'formal_charge':
    list(range(-5, 7)),
    'num_hs':
    list(range(0, 9)),
    'num_radical_electrons':
    list(range(0, 5)),
    'hybridization': [
        'UNSPECIFIED',
        'S',
        'SP',
        'SP2',
        'SP3',
        'SP3D',
        'SP3D2',
        'OTHER',
    ],
    'is_aromatic': [False, True],
    'is_in_ring': [False, True],
}

e_map = {
    'bond_type': [
        'UNSPECIFIED',
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'QUADRUPLE',
        'QUINTUPLE',
        'HEXTUPLE',
        'ONEANDAHALF',
        'TWOANDAHALF',
        'THREEANDAHALF',
        'FOURANDAHALF',
        'FIVEANDAHALF',
        'AROMATIC',
        'IONIC',
        'HYDROGEN',
        'THREECENTER',
        'DATIVEONE',
        'DATIVE',
        'DATIVEL',
        'DATIVER',
        'OTHER',
        'ZERO',
    ],
    'stereo': [
        'STEREONONE',
        'STEREOANY',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
    ],
    'is_conjugated': [False, True],
}



def from_mol(mol):
    """
    Converts a mol to a :class:`torch_geometric.data.Data` instance.

    :param mol: an rdkit molecule
    :return: Data
    """

    xs = []
    for atom in mol.GetAtoms():
        x = []
        x.append(x_map['atomic_num'].index(atom.GetAtomicNum()))
        x.append(x_map['chirality'].index(str(atom.GetChiralTag())))
        x.append(x_map['degree'].index(atom.GetTotalDegree()))
        x.append(x_map['formal_charge'].index(atom.GetFormalCharge()))
        x.append(x_map['num_hs'].index(atom.GetTotalNumHs()))
        x.append(x_map['num_radical_electrons'].index(
            atom.GetNumRadicalElectrons()))
        x.append(x_map['hybridization'].index(str(atom.GetHybridization())))
        x.append(x_map['is_aromatic'].index(atom.GetIsAromatic()))
        x.append(x_map['is_in_ring'].index(atom.IsInRing()))
        xs.append(x)

    x = torch.tensor(xs, dtype=torch.long).view(-1, 9)

    edge_indices, edge_attrs = [], []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        e = []
        e.append(e_map['bond_type'].index(str(bond.GetBondType())))
        e.append(e_map['stereo'].index(str(bond.GetStereo())))
        e.append(e_map['is_conjugated'].index(bond.GetIsConjugated()))

        edge_indices += [[i, j], [j, i]]
        edge_attrs += [e, e]

    edge_index = torch.tensor(edge_indices)
    edge_index = edge_index.t().to(torch.long).view(2, -1)
    edge_attr = torch.tensor(edge_attrs, dtype=torch.long).view(-1, 3)

    if edge_index.numel() > 0:  # Sort indices.
        perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
        edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]

    return {'x': x, 'edge_index': edge_index, 'edge_attr': edge_attr}

# functions to convert pyg Data to the format used by graphormer
# these use Cython functions in algos.pyx, so need to do "Cython algos.pyx"

@torch.jit.script
def convert_to_single_emb(x, offset: int = 512):
    feature_num = x.size(1) if len(x.size()) > 1 else 1
    feature_offset = 1 + torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    x = x + feature_offset
    return x


def preprocess_item(item):
    edge_attr, edge_index, x = item['edge_attr'], item['edge_index'], item['x']
    N = x.size(0)
    x = convert_to_single_emb(x)

    # node adj matrix [N, N] bool
    adj = torch.zeros([N, N], dtype=torch.bool)
    adj[edge_index[0, :], edge_index[1, :]] = True

    # edge feature here
    if len(edge_attr.size()) == 1:
        edge_attr = edge_attr[:, None]
    attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.long)
    attn_edge_type[edge_index[0, :], edge_index[1, :]] = (
        convert_to_single_emb(edge_attr) + 1
    )

    shortest_path_result, path = algos.floyd_warshall(adj.numpy())
    max_dist = np.amax(shortest_path_result)
    edge_input = algos.gen_edge_input(max_dist, path, attn_edge_type.numpy())
    spatial_pos = torch.from_numpy((shortest_path_result)).long()
    attn_bias = torch.zeros([N + 1, N + 1], dtype=torch.float)  # with graph token

    # combine
    return_item = {
        'x': x,
        'attn_bias': attn_bias,
        'attn_edge_type': attn_edge_type,
        'spatial_pos': spatial_pos,
        'in_degree': adj.long().sum(dim=1).view(-1),
        'out_degree': adj.long().sum(dim=1).view(-1),
        'edge_input': torch.from_numpy(edge_input).long(),
    }

    return return_item


class EmbedGraphormerMol(Embed):
    """
    Embedding of mol for Graphormer
    """

    def __init__(self, config):
        super().__init__(config)

    def mol_embed(self, row):
        """
        embed a processed mol

        :param row: data record
        :return: one hot tensor
        """

        item = None
        if row['mol'] is not None:
            try:
                item = from_mol(row['mol'])
                item = preprocess_item(item)
            except ValueError as e:
                logging.info(f'{e}: unable to create embedding for {row}')
        
        return item

    @staticmethod
    def mol_channels(self):
        """
        the number of mol channels

        :return: the number of mol channels
        """
        return 1

    def embed(self, row):
        """
        call the requested embedding functions as listed in config.ml.embedding.embeddings

        :param row: the data row
        :return: the concatenated one hot tensor of the embeddings
        """
        try:
            embeddings = [
                getattr(self, func + "_embed")(row)
                for func in self.config.ml.embedding.embeddings
            ]
        except KeyError as e:
            logging.error(f"not able to find embedding: {e}")
            raise
        # todo: this is hardcoded to just return the mol embedding.  Needs to be a more abstract way to do this
        return embeddings[0]


class EmbedPATNMol(Embed):
    """
    Embedding for PATN PropPredictor
    """

    def __init__(self, config):
        super().__init__(config)

    def mol_path_embed(self, row):
        """
        embed the nce as a one hot tensor

        :param row: data record
        :return: one hot tensor
        """
        mol = row['mol']
        n_atoms = mol.GetNumAtoms()

        path_input, path_mask = path_utils.get_path_input(
            [mol], [row['shortest_paths']], n_atoms, self.config.ml.model.PropPredictor, output_tensor=False)
        path_input = path_input.squeeze(0)  # Remove batch dimension
        path_mask = path_mask.squeeze(0)  # Remove batch dimension

        return mol, n_atoms, path_input, path_mask

    @staticmethod
    def mol_channels(self):
        """
        the number of mol channels

        :return: the number of mol channels
        """
        return 1

    def embed(self, row):
        """
        call the requested embedding functions as listed in config.ml.embedding.embeddings

        :param row: the data row
        :return: the concatenated one hot tensor of the embeddings
        """
        try:
            embeddings = [
                getattr(self, func + "_embed")(row)
                for func in self.config.ml.embedding.embeddings
            ]
        except KeyError as e:
            logging.error(f"not able to find embedding: {e}")
            raise
        # todo: this is hardcoded to just return the mol embedding.  Needs to be a more abstract way to do this
        return embeddings[0]
