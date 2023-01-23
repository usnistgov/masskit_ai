import torch
import torch.nn as nn
from rdkit import Chem
from masskit_ai.mol.small.models import path_utils
from masskit_ai.mol.small.models import mol_features
from masskit_ai.mol.small.models  import model_utils
from masskit_ai.spectrum.spectrum_base_objects import SpectrumModel
from masskit_ai.base_objects import ModelOutput


def __getitem__(self, index):
    smiles, label = self.data[index]
    mol = Chem.MolFromSmiles(smiles)
    n_atoms = mol.GetNumAtoms()

    path_input = None
    path_mask = None
    # if self.args.use_paths:
    shortest_paths = [self.args.p_info[smiles]]
    path_input, path_mask = path_utils.get_path_input(
        [mol], shortest_paths, n_atoms, self.args, output_tensor=False)
    path_input = path_input.squeeze(0)  # Remove batch dimension
    path_mask = path_mask.squeeze(0)  # Remove batch dimension
    return smiles, label, n_atoms, (path_input, path_mask)

"""
# batch data is from getitem
        smiles_list, labels_list, path_tuple = batch_data
        path_input, path_mask = path_tuple
        if args.use_paths:
            path_input = path_input.to(args.device)
            path_mask = path_mask.to(args.device)

        n_data = len(smiles_list)
        # note that MolGraph is a list.
        mol_graph = MolGraph(smiles_list, args, path_input, path_mask)

        pred_logits = prop_predictor(mol_graph, stats_tracker).squeeze(1)
        labels = torch.tensor(labels_list, device=args.device)


- put above code into a special purpose collate_fn

"""

class PropPredictor(SpectrumModel):
    def __init__(self, args, n_classes=1):
        super(PropPredictor, self).__init__(args)
        self.args = args.ml.model.PropPredictor
        hidden_size = self.args.hidden_size

        self.model = MolTransformer(self.args)

        self.W_p_h = nn.Linear(self.model.output_size, hidden_size)  # Prediction
        self.W_p_o = nn.Linear(hidden_size, n_classes)

    def aggregate_atom_h(self, atom_h, scope):
        mol_h = []
        for (st, le) in scope:
            cur_atom_h = atom_h.narrow(0, st, le)

            if self.args.agg_func == 'sum':
                mol_h.append(cur_atom_h.sum(dim=0))
            elif self.args.agg_func == 'mean':
                mol_h.append(cur_atom_h.mean(dim=0))
            else:
                assert(False)
        mol_h = torch.stack(mol_h, dim=0)
        return mol_h

    def forward(self, input, output_attn=False):
        mol_graph = input[0]
        attn_list = None

        # move tensors to cuda device.  Not placed on cuda device in dataloaders as this doesn't work from
        # forked processes
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        mol_graph.path_input = mol_graph.path_input.to(device=device)
        mol_graph.path_mask = mol_graph.path_mask.to(device=device)

        atom_h, attn_list = self.model(mol_graph)

        scope = mol_graph.scope
        mol_h = self.aggregate_atom_h(atom_h, scope)
        mol_h = nn.ReLU()(self.W_p_h(mol_h))
        mol_o = self.W_p_o(mol_h)
        mol_o = mol_o.squeeze(-1)

        if not output_attn:
            return ModelOutput(mol_o)
        else:
            return mol_o, attn_list


class MolTransformer(nn.Module):
    def __init__(self, args):
        super(MolTransformer, self).__init__()
        self.args = args
        hidden_size = args.hidden_size
        n_heads, d_k = args.n_heads, args.d_k

        n_atom_feats = mol_features.N_ATOM_FEATS
        n_path_feats = path_utils.get_num_path_features(args)

        # W_atom_i: input atom embedding
        self.W_atom_i = nn.Linear(n_atom_feats, n_heads * d_k, bias=False)

        # W_attn_h: compute atom attention score
        # W_message_h: compute the new atom embeddings
        n_score_feats = 2 * d_k + n_path_feats
        if args.no_share:
            self.W_attn_h = nn.ModuleList([
                nn.Linear(n_score_feats, d_k) for _ in range(args.depth - 1)])
            self.W_attn_o = nn.ModuleList([
                nn.Linear(d_k, 1) for _ in range(args.depth - 1)])
            self.W_message_h = nn.ModuleList([
                nn.Linear(n_score_feats, d_k) for _ in range(args.depth - 1)])
        else:
            self.W_attn_h = nn.Linear(n_score_feats, d_k)
            self.W_attn_o = nn.Linear(d_k, 1)
            self.W_message_h = nn.Linear(n_score_feats, d_k)

        # W_atom_o: the output embedding
        self.W_atom_o = nn.Linear(n_atom_feats + n_heads * d_k, hidden_size)
        self.dropout = nn.Dropout(args.dropout)

        self.output_size = hidden_size

    def get_attn_input(self, atom_h, path_input, max_atoms):
        # attn_input is concatentation of atom pair embeddings and path input
        atom_h1 = atom_h.unsqueeze(2).expand(-1, -1, max_atoms, -1)
        atom_h2 = atom_h.unsqueeze(1).expand(-1, max_atoms, -1, -1)
        atom_pairs_h = torch.cat([atom_h1, atom_h2], dim=3)
        attn_input = torch.cat([atom_pairs_h, path_input], dim=3)

        return attn_input

    def compute_attn_probs(self, attn_input, attn_mask, layer_idx, eps=1e-20):
        # attn_scores is [batch, atoms, atoms, 1]
        if self.args.no_share:
            attn_scores = nn.LeakyReLU(0.2)(
                self.W_attn_h[layer_idx](attn_input))
            attn_scores = self.W_attn_o[layer_idx](attn_scores) * attn_mask
        else:
            attn_scores = nn.LeakyReLU(0.2)(
                self.W_attn_h(attn_input))
            attn_scores = self.W_attn_o(attn_scores) * attn_mask

        # max_scores is [batch, atoms, 1, 1], computed for stable softmax
        max_scores = torch.max(attn_scores, dim=2, keepdim=True)[0]
        # exp_attn is [batch, atoms, atoms, 1]
        exp_attn = torch.exp(attn_scores - max_scores) * attn_mask
        # sum_exp is [batch, atoms, 1, 1], add eps for stability
        sum_exp = torch.sum(exp_attn, dim=2, keepdim=True) + eps

        # attn_probs is [batch, atoms, atoms, 1]
        attn_probs = (exp_attn / sum_exp) * attn_mask
        return attn_probs

    def avg_attn(self, attn_probs, n_heads, batch_sz, max_atoms):
        if n_heads > 1:
            attn_probs = attn_probs.view(n_heads, batch_sz, max_atoms, max_atoms)
            attn_probs = torch.mean(attn_probs, dim=0)
        return attn_probs

    def forward(self, mol_graph):
        atom_input, scope = mol_graph.get_atom_inputs()
        max_atoms = model_utils.compute_max_atoms(scope)
        atom_input_3D, atom_mask = model_utils.convert_to_3D(
            atom_input, scope, max_atoms, self.args.self_attn)
        path_input, path_mask = mol_graph.path_input, mol_graph.path_mask

        batch_sz, _, _ = atom_input_3D.size()
        n_heads, d_k = self.args.n_heads, self.args.d_k

        # Atom mask allows all valid atoms
        # Path mask allows only atoms in the neighborhood
        if self.args.mask_neigh:
            attn_mask = path_mask
        else:
            attn_mask = atom_mask.float()
        attn_mask = attn_mask.unsqueeze(3)

        if n_heads > 1:
            attn_mask = attn_mask.repeat(n_heads, 1, 1, 1)
            path_input = path_input.repeat(n_heads, 1, 1, 1)
            path_mask = path_mask.repeat(n_heads, 1, 1)  # Used to compute neighbor score

        atom_input_h = self.W_atom_i(atom_input_3D).view(batch_sz, max_atoms, n_heads, d_k)
        atom_input_h = atom_input_h.permute(2, 0, 1, 3).contiguous().view(-1, max_atoms, d_k)

        attn_list = []

        # atom_h should be [batch_size * n_heads, atoms, # features]
        atom_h = atom_input_h
        for layer_idx in range(self.args.depth - 1):
            attn_input = self.get_attn_input(atom_h, path_input, max_atoms)

            attn_probs = self.compute_attn_probs(attn_input, attn_mask, layer_idx)
            attn_list.append(self.avg_attn(attn_probs, n_heads, batch_sz, max_atoms))
            attn_probs = self.dropout(attn_probs)

            if self.args.no_share:
                attn_h = self.W_message_h[layer_idx](
                    torch.sum(attn_probs * attn_input, dim=2))
            else:
                attn_h = self.W_message_h(
                    torch.sum(attn_probs * attn_input, dim=2))
            atom_h = nn.ReLU()(attn_h + atom_input_h)

        # Concat heads
        atom_h = atom_h.view(n_heads, batch_sz, max_atoms, -1)
        atom_h = atom_h.permute(1, 2, 0, 3).contiguous().view(batch_sz, max_atoms, -1)

        atom_h = model_utils.convert_to_2D(atom_h, scope)
        atom_output = torch.cat([atom_input, atom_h], dim=1)
        atom_h = nn.ReLU()(self.W_atom_o(atom_output))

        return atom_h, attn_list
