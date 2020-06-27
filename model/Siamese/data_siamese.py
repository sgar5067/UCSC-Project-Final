import config as ec
from data import Data
from coarsening import coarsen, perm_data
from utils_siamese import get_coarsen_level
from utils import load_data, exec_turnoff_print, exec_turnon_print
from node_ordering import node_ordering
from random_walk_generator import generate_random_walks
from supersource_generator import generate_supersource
import scipy.sparse as sp
from sklearn.preprocessing import OneHotEncoder
import networkx as nx
import numpy as np

# exec_turnoff_print()
exec_turnon_print()

logging_enabled = True

class SiameseModelData(Data):
    """Represents a set of data for training a Siamese model.

    This object is saved to disk, so every time preprocessing is changed, the saved files should
    be archived and re-run so that the next training run uses the updated preprocessing. Each
    model data is uniquely identified by the dataset name and FLAGS settings. This object should
    contain everything required as data input that can be precomputed. The saved binary file's
    name has all parameters that uniquely identify it. The main objects representing the graphs
    are ModelGraph objects.
    
    """

    if logging_enabled == True:
        print("- Entered data_siamese::SiameseModelData Class")

    def __init__(self, dataset):
        if logging_enabled == True:
            print("- Entered data_siamese::SiameseModelData::__init__ Constructor Method")

        # The attributes set below determine the save file name.
        self.dataset = dataset
        self.valid_percentage = ec.valid_percentage
        self.node_feat_name = ec.node_feat_name
        self.node_feat_encoder = ec.node_feat_encoder
        self.ordering = ec.ordering
        self.coarsening = ec.coarsening
        self.supersource = ec.supersource
        self.random_walk = ec.random_walk
        self.laplacian = ec.laplacian

        super().__init__(self.get_name())
        
        print('{} train graphs; {} validation graphs; {} test graphs'.format(
            len(self.train_gs),
            len(self.val_gs),
            len(self.test_gs)))

    def init(self):
        """Creates the object from scratch but only if a saved version doesn't already exist."""

        if logging_enabled == True:
            print("- Entered data_siamese::SiameseModelData::init Public Method")

        orig_train_data = load_data(self.dataset, train=True)
        train_gs, val_gs = self._train_val_split(orig_train_data)
        test_gs = load_data(self.dataset, train=False).graphs

        # Note that <graph> and self.<graph> can have different graphs because of the supersource
        # option. This turns the graph into a DiGraph and adds a node, so the graph is
        # fundamentally changed with the supersource setting. Use self.<graph> as truth.
        
        self.node_feat_encoder = self._create_node_feature_encoder(orig_train_data.graphs + test_gs)
        self.graph_label_encoder = GraphLabelOneHotEncoder(orig_train_data.graphs + test_gs)
        
        self._check_graphs_num(test_gs, 'test')
        self.train_gs = self.create_model_gs(train_gs, 'train')
        self.val_gs = self.create_model_gs(val_gs, 'val')
        self.test_gs = self.create_model_gs(test_gs, 'test')
        
        assert (len(train_gs) + len(val_gs) == len(orig_train_data.graphs))

    def input_dim(self):
        if logging_enabled == True:
            print("- Entered data_siamese::SiameseModelData::input_dim Public Method")

        return self.node_feat_encoder.input_dim()

    def num_graphs(self):
        if logging_enabled == True:
            print("- Entered data_siamese::SiameseModelData::num_graphs Public Method")
        
        return len(self.train_gs) + len(self.val_gs) + len(self.test_gs)

    def create_model_gs(self, gs, tvt):
        if logging_enabled == True:
            print("- Entered data_siamese::SiameseModelData::create_model_gs Public Method")
        
        rtn = []
        hits = [0, 0.3, 0.6, 0.9]
        cur_hit = 0
        
        for i, g in enumerate(gs):
            mg = ModelGraph(g, self.node_feat_encoder, self.graph_label_encoder)
            perc = i / len(gs)
            if cur_hit < len(hits) and abs(perc - hits[cur_hit]) <= 0.05:
                print('{} {}/{}={:.1%}'.format(tvt, i, len(gs), i / len(gs)))
                cur_hit += 1
            rtn.append(mg)
        
        return rtn

    def get_name(self):
        if logging_enabled == True:
            print("- Entered data_siamese::SiameseModelData::get_name Public Method")
        
        if hasattr(self, 'name'):
            return self.name
        
        else:
            li = []
            for k, v in sorted(self.__dict__.items(), key=lambda x: x[0]):
                li.append('{}'.format(v))
            self.name = '_'.join(li)
            
        return self.name

    def _train_val_split(self, orig_train_data):
        if logging_enabled == True:
            print("- Entered data_siamese::SiameseModelData::_train_val_split Private Method")
        
        if self.valid_percentage < 0 or self.valid_percentage > 1:
            raise RuntimeError('valid_percentage {} must be in [0, 1]'.format(self.valid_percentage))
        
        gs = orig_train_data.graphs
        sp = int(len(gs) * (1 - self.valid_percentage))
        train_graphs = gs[0:sp]
        valid_graphs = gs[sp:]
        self._check_graphs_num(train_graphs, 'train')
        self._check_graphs_num(valid_graphs, 'validation')
        
        return train_graphs, valid_graphs

    def _check_graphs_num(self, graphs, label):
        if logging_enabled == True:
            print("- Entered data_siamese::SiameseModelData::_check_graphs_num Private Method")
        
        if len(graphs) <= 2:
            raise RuntimeError('Insufficient {} graphs {}'.format(label, len(graphs)))

    def _create_node_feature_encoder(self, gs):
        if logging_enabled == True:
            print("- Entered data_siamese::SiameseModelData::_create_node_feature_encoder Private Method")
        
        if self.node_feat_encoder == 'onehot':
            return NodeFeatureOneHotEncoder(gs, self.node_feat_name)
        
        elif 'constant' in self.node_feat_encoder:
            return NodeFeatureConstantEncoder(gs, self.node_feat_name)
        
        else:
            raise RuntimeError('Unknown node_feat_encoder {}'.format(
                self.node_feat_encoder))

class ModelGraph(object):
    """Defines all relevant graph properties required for training a Siamese model.

    This is a data model representation of a graph for use during the training stage.
    Each ModelGraph has precomputed parameters (Laplacian, inputs, adj matrix, etc) as needed by the
    network during training.
    """

    if logging_enabled == True:
        print("- Entered data_siamese::ModelGraph Class")

    def __init__(self, nxgraph, node_feat_encoder, graph_label_encoder):
        if logging_enabled == True:
            print("- Entered data_siamese::ModelGraph::__init__ Constructor Method")
        
        self.glabel_position = graph_label_encoder.encode(nxgraph)
        
        # Check flag compatibility.
        self._error_if_incompatible_flags()

        self.nxgraph = nxgraph
        last_order = []

        # Generates random walks with parameters determined by the flags. Walks are defined
        # by the ground truth node ids, so they do not depend on ordering, but if a supersource
        # node is used it should be generated before the random walk.
        
        if ec.random_walk:
            params = ec.random_walk.split('_')
            num_walks = int(params[0])
            walk_length = int(params[1])
            self.random_walk_data = generate_random_walks(nxgraph, num_walks, walk_length)

        # Encode features.
        dense_node_inputs = node_feat_encoder.encode(nxgraph)
        
        # Determine ordering and reorder the dense inputs
        # based on the desired ordering.
        if ec.ordering:
            if ec.ordering == 'bfs':
                self.order, self.mapping = node_ordering(
                    nxgraph, 'bfs', ec.node_feat_name, last_order)

            elif ec.ordering == 'degree':
                self.order, self.mapping = node_ordering(
                    nxgraph, 'degree', ec.node_feat_name, last_order)
            
            else:
                raise RuntimeError('Unknown ordering mode {}'.format(self.order))
            
            assert (len(self.order) == len(nxgraph.nodes()))
            # Apply the ordering.
            dense_node_inputs = dense_node_inputs[self.order, :]

        # Save matrix properties after reordering the nodes.
        self.sparse_node_inputs = self._preprocess_inputs(sp.csr_matrix(dense_node_inputs))
        
        # Only one laplacian.
        self.num_laplacians = 1
        if nxgraph.number_of_nodes() < 500000:
            adj = nx.to_numpy_matrix(nxgraph)
        else:
            adj = nx.to_scipy_sparse_matrix(nxgraph)
        
        # Fix ordering for adj.
        if ec.ordering:
            # Reorders the adj matrix using the order provided earlier.
            adj = adj[np.ix_(self.order, self.order)]

        # Special handling for coarsening because it is
        # incompatible with other flags.
        if ec.coarsening:
            self._coarsen(dense_node_inputs, adj)
        else:
            self.laplacians = [self._preprocess_adj(adj)]

    def _error_if_incompatible_flags(self):
        """Check flags and error for unhandled flag combinations.
            ec.coarsening
            ec.ordering
            ec.supersource
            ec.random_walk
        """
        if logging_enabled == True:
            print("- Entered data_siamese::ModelGraph::_error_if_incompatible_flags Private Method")
        
        if ec.coarsening:
            if ec.ordering or ec.supersource or ec.random_walk:
                raise RuntimeError(
                    'Cannot use coarsening with any of the following: ordering, '
                    'supersource, random_walk')

    def get_nxgraph(self):
        if logging_enabled == True:
            print("- Entered data_siamese::ModelGraph::get_nxgraph Public Method")
        
        return self.nxgraph

    def get_node_inputs(self):
        if logging_enabled == True:
            print("- Entered data_siamese::ModelGraph::get_node_inputs Public Method")
        
        if ec.coarsening:
            return self.sparse_permuted_padded_dense_node_inputs
        else:
            return self.sparse_node_inputs

    def get_node_inputs_num_nonzero(self):
        if logging_enabled == True:
            print("- Entered data_siamese::ModelGraph::get_node_inputs_num_nonzero Public Method")
        
        return self.get_node_inputs()[1].shape

    def get_laplacians(self, gcn_id):
        if logging_enabled == True:
            print("- Entered data_siamese::ModelGraph::get_laplacians Public Method")
        
        if ec.coarsening:
            return self.coarsened_laplacians[gcn_id]
        else:
            return self.laplacians

    def _coarsen(self, dense_node_inputs, adj):
        if logging_enabled == True:
            print("- Entered data_siamese::ModelGraph::_coarsen Private Method")
        
        assert ('metis_' in ec.coarsening)
        self.num_level = get_coarsen_level()
        assert (self.num_level >= 1)
        graphs, perm = coarsen(sp.csr_matrix(adj), levels=self.num_level, self_connections=False)
        permuted_padded_dense_node_inputs = perm_data(dense_node_inputs.T, perm).T
        self.sparse_permuted_padded_dense_node_inputs = self._preprocess_inputs(sp.csr_matrix(permuted_padded_dense_node_inputs))
        self.coarsened_laplacians = []
        
        for g in graphs:
            self.coarsened_laplacians.append([self._preprocess_adj(g.todense())])
        
        assert (len(self.coarsened_laplacians) == self.num_laplacians * self.num_level + 1)

    def _preprocess_inputs(self, inputs):
        """Row-normalize feature matrix and convert to tuple representation"""
        
        if logging_enabled == True:
            print("- Entered data_siamese::ModelGraph::_preprocess_inputs Private Method")
        
        rowsum = np.array(inputs.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        inputs = r_mat_inv.dot(inputs)
        
        return self._sparse_to_tuple(inputs)

    def _preprocess_adj(self, adj):
        """Preprocessing of adjacency matrix and conversion to tuple representation."""
        
        if logging_enabled == True:
            print("- Entered data_siamese::ModelGraph::_preprocess_adj Private Method")
        
        adj_proc = sp.coo_matrix(adj + sp.eye(adj.shape[0]))
        
        if ec.laplacian == 'gcn':
            adj_proc = self._normalize_adj(adj_proc)
        
        self._edge_list_incidence_mat(adj)
        
        return self._sparse_to_tuple(adj_proc)

    def _edge_list_incidence_mat(self, adj):
        if logging_enabled == True:
            print("- Entered data_siamese::ModelGraph::_edge_list_incidence_mat Private Method")
        
        tmp_G = nx.from_numpy_matrix(adj)
        edge_list = tmp_G.edges()
        edge_list_full = []
        
        for (i, j) in edge_list:
            edge_list_full.append((i, j))
            if i != j:
                edge_list_full.append((j, i))
        
        self.edge_index = self._sparse_to_tuple(sp.csr_matrix(edge_list_full))
        incidence_mat = self._our_incidence_mat(tmp_G, edgelist=edge_list_full)
        self.incidence_mat = self._sparse_to_tuple(incidence_mat)

    def _our_incidence_mat(self, G, edgelist):
        if logging_enabled == True:
            print("- Entered data_siamese::ModelGraph::_our_incidence_mat Private Method")
        
        nodelist = G.nodes()
        A = sp.lil_matrix((len(nodelist), len(edgelist)))
        node_index = dict((node, i) for i, node in enumerate(nodelist))
        
        for ei, e in enumerate(edgelist):
            (u, v) = e[:2]
            if u == v: continue  # self loops give zero column
            ui = node_index[u]
            # vi = node_index[v]
            A[ui, ei] = 1
        
        return A.asformat('csc')

    def _normalize_adj(self, adj):
        """Symmetrically normalize adjacency matrix."""
        
        if logging_enabled == True:
            print("- Entered data_siamese::ModelGraph::_normalize_adj Private Method")
        
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

    def _sparse_to_tuple(self, sparse_mx):
        """Convert sparse matrix to tuple representation."""

        if logging_enabled == True:
            print("- Entered data_siamese::ModelGraph::_sparse_to_tuple Private Method")

        def to_tuple(mx):
            if not sp.isspmatrix_coo(mx):
                mx = mx.tocoo()
                
            coords = np.vstack((mx.row, mx.col)).transpose()
            values = mx.data
            shape = mx.shape
            return coords, values, shape

        if isinstance(sparse_mx, list):
            for i in range(len(sparse_mx)):
                sparse_mx[i] = to_tuple(sparse_mx[i])
        else:
            sparse_mx = to_tuple(sparse_mx)

        return sparse_mx

class NodeFeatureEncoder(object):
    if logging_enabled == True:
        print("- Entered data_siamese::NodeFeatureEncoder Class")

    def encode(self, g):
        raise NotImplementedError()

    def input_dim(self):
        raise NotImplementedError()

class NodeFeatureOneHotEncoder(NodeFeatureEncoder):
    if logging_enabled == True:
        print("- Entered data_siamese::NodeFeatureOneHotEncoder Class")

    def __init__(self, gs, node_feat_name):
        if logging_enabled == True:
            print("- Entered data_siamese::NodeFeatureOneHotEncoder::__init__ Constructor Method")
        
        self.node_feat_name = node_feat_name
        # Go through all the graphs in the entire dataset
        # and create a set of all possible
        # labels so we can one-hot encode them.
        inputs_set = set()
        
        for g in gs:
            inputs_set = inputs_set | set(self._node_feat_dic(g).values())
                
        self.feat_idx_dic = {feat: idx for idx, feat in enumerate(inputs_set)}
        self._fit_onehotencoder()

    def _fit_onehotencoder(self):
        if logging_enabled == True:
            print("- Entered data_siamese::NodeFeatureOneHotEncoder::_fit_onehotencoder Private Method")
        
        keys = self.feat_idx_dic.keys()
        for key in keys:
            value = self.feat_idx_dic.get(key)
                   
        self.oe = OneHotEncoder().fit(np.array(list(self.feat_idx_dic.values())).reshape(-1, 1))

    def add_new_feature(self, feat_name):
        """Use this function if a new feature was added to the graph set."""
        
        if logging_enabled == True:
            print("- Entered data_siamese::NodeFeatureOneHotEncoder::add_new_feature Public Method")
        
        # Add the new feature to the dictionary as a unique feature and reinit the encoder.
        new_idx = len(self.feat_idx_dic)
        self.feat_idx_dic[feat_name] = new_idx
        self._fit_onehotencoder()

    def encode(self, g):
        if logging_enabled == True:
            print("- Entered data_siamese::NodeFeatureOneHotEncoder::encode Public Method")
        
        node_feat_dic = self._node_feat_dic(g)
        temp = [self.feat_idx_dic[node_feat_dic[n]] for n in g.nodes()]
        
        return self.oe.transform(np.array(temp).reshape(-1, 1)).toarray()

    def input_dim(self):
        if logging_enabled == True:
            print("- Entered data_siamese::NodeFeatureOneHotEncoder::input_dim Public Method")
                
        return self.oe.transform([[0]]).shape[1]

    def _node_feat_dic(self, g):
        if logging_enabled == True:
            print("- Entered data_siamese::NodeFeatureOneHotEncoder::_node_feat_dic Private Method")
        
        return nx.get_node_attributes(g, name=self.node_feat_name)


class NodeFeatureConstantEncoder(NodeFeatureEncoder):
    if logging_enabled == True:
        print("- Entered data_siamese::NodeFeatureConstantEncoder Class")
    
    def __init__(self, _, node_feat_name):
        if logging_enabled == True:
            print("- Entered data_siamese::NodeFeatureConstantEncoder::__init__ Constructor Method")
        
        self.input_dim_ = int(ec.node_feat_encoder.split('_')[1])
        self.const = float(2.0)
        assert (node_feat_name is None)

    def encode(self, g):
        if logging_enabled == True:
            print("- Entered data_siamese::NodeFeatureConstantEncoder::encode Public Method")
        
        rtn = np.full((g.number_of_nodes(), self.input_dim_), self.const)        
        return rtn

    def input_dim(self):
        if logging_enabled == True:
            print("- Entered data_siamese::NodeFeatureConstantEncoder::input_dim Public Method")
        
        return self.input_dim_

class GraphLabelOneHotEncoder(object):
    if logging_enabled == True:
        print("- Entered data_siamese::GraphLabelOneHotEncoder Class")
    
    def __init__(self, gs):
        if logging_enabled == True:
            print("- Entered data_siamese::GraphLabelOneHotEncoder::__init__ Constructor Method")
        
        self.glabel_map = {}
        for g in gs:
            glabel = g.graph['glabel']
            if glabel not in self.glabel_map:
                self.glabel_map[glabel] = len(self.glabel_map)

    def encode(self, g):
        if logging_enabled == True:
            print("- Entered data_siamese::GraphLabelOneHotEncoder::encode Public Method")
        
        return self.glabel_map[g.graph['glabel']]
