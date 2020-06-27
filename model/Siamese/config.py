#####################################
# configs attempted:
#     original: coarsening = None, ordering = 'bfs', ds_kernal = 'exp', lambda_msel = 1, learning_rate = .001, weight_decay = 0,
#               pred_sim_dist = 'sim', lambda_wdl = 0,
#     1. coarsening = 0, ordering = None, result: clustering seems a bit tighter then without coarsening
#     2. coarsening = 0, ordering = None, ds_kernal = 'gaussian', result: I can see clusters but seems a bit off
#     3. coarsening = None, ordering = 'bfs', ds_kernal = 'gaussian', result: best by far so far
#     4. coarsening = None, ordering = 'bfs', ds_kernal = 'gaussian', lambda_msel = {1, .1, .01, .001, .0001}, 
#        result: changes to lambda were not good, should remain at 1
#     5. coarsening = None, ordering = 'bfs', ds_kernal = 'gaussian', lambda_msel = 1, 
#        learning_rate = {.1 - fails, .01 - NG, .001 - G, .0001 - NG, .00001 - NG}
#        result: changes to learning_rate were not good, should remain at .001
#     6. coarsening = None, ordering = 'bfs', ds_kernal = 'gaussian', lambda_msel = 1, learning_rate = .001, 
#        weight_decay = {0 - G, .1 - VG, .01 - G, .001 - VB}
#     7. coarsening = None, ordering = 'bfs', ds_kernal = 'gaussian', lambda_msel = 0, lambda_wdl = 1, learning_rate = .001, 
#        pred_sim_dist = 'dist'
#        result: I didn't try lambda_tsl or _del and the attempt at _wdl didn't work due to pred_sim_dist and supply_sim_dist values. 
#                Keep lambda_msel
#####################################



#####################################
###### Data Preprocessing.
#####################################

dataset = 'project'
# dataset = 'ptc'

dataset_train = dataset
dataset_val_test = dataset
if 'aids' in dataset or dataset in ['webeasy', 'nci109', 'ptc', 'project']:
    # node_feat_name = 'node_name'
    node_feat_name = 'type'
    node_feat_encoder = 'onehot'
    max_nodes = 10
    if dataset == 'webeasy':
        max_nodes = 404
        num_glabels = 20
    if dataset == 'nci109':
        max_nodes = 106
        num_glabels = 2
    if dataset == 'ptc':
        max_nodes = 109
        num_glabels = 2
    if dataset == 'project':
        max_nodes = 107
        num_glabels = 8
elif 'imdb' in dataset:
    node_feat_name = None
    node_feat_encoder = 'constant_1'
    max_nodes = 90
    num_glabels = 3

dataset_train = dataset_train
dataset_val_test = dataset_val_test
num_glabels = num_glabels
node_feat_name = node_feat_name
node_feat_encoder = node_feat_encoder

# valid_percentage: (0, 1), (# of validation graphs / (# of validation + # of training graphs)
valid_percentage = 0.20

# distance similarity metric
ds_metric = 'ged'

# Ground-truth distance algorithm to use
ds_algo = 'astar'

# ordering: 'bfs', 'degree', None
# ordering = 'bfs'
ordering = 'bfs'

# Algorithm for graph coarsening --> coarsening: 'metis_<num_level>' None.
# coarsening = None
coarsening = None
laplacian = 'gcn'

#####################################
###### Model Config
#####################################
model = 'siamese_regression'
model_name = 'Our Model'

# Number of graph pairs in a batch
batch_size = 2

# Whether to normalize the distance or not when choosing the ground truth distance
ds_norm = True

# Whether to normalize the node embeddings or not
node_embs_norm = False

pred_sim_dist = None
supply_sim_dist = None

# ds_kernel: gaussian, exp, inverse or identity
# ds_kernel = 'exp'
ds_kernel = 'gaussian'
if ds_metric == 'glet':
    ds_kernel = 'identity'

# Name of the similarity kernel
yeta = None
scale = None

if ds_kernel == 'gaussian':
    # yeta - gaussian kernel function
    # if ds_norm, try 0.6 for nef small, 0.3 for nef, 0.2 for regular.
    # else, try 0.01 for nef, 0.001 for regular
    yeta = 0.01

elif ds_kernel == 'exp' or ds_kernel == 'inverse':
    # Scale for the exp/inverse kernel function
    scale = 0.7
    
pred_sim_dist = 'sim'

if ds_metric == 'mcs' or ds_metric == 'glet':
    pred_sim_dist = 'sim'

supply_sim_dist = pred_sim_dist

#####################################
###### MSE loss Config
#####################################

lambda_msel = 1  # 1  # 1 #0.0001
lambda_mse_loss = 0

if lambda_msel > 0:    
    lambda_mse_loss = lambda_msel

#####################################
###### Weighted Distance Loss Config
#####################################

lambda_wdl = 0  # 1#1  # 1
lambda_weighted_dist_loss = 0

if lambda_wdl > 0:
    lambda_weighted_dist_loss = lambda_wdl
    
    #  special for wdl loss
    supply_sim_dist = 'sim'
    
    # Graph Embedding normalization
    # graph_embs_norm = True

#######################################
###### Trivial Solution Avoidance Loss
#######################################

lambda_tsl = 0
lambda_triv_avoid_loss = 0

# Lambda for the trivial solution avoidance loss
if lambda_tsl > 0:
    lambda_triv_avoid_loss = lambda_tsl
    
#######################################
###### Diversity Encouraging Loss
#######################################

lambda_del = 0
lambda_diversity_loss = 0

# Lambda for the diversity encouraging loss
if lambda_del > 0:
    lambda_diversity_loss = lambda_del

########################################


# dist/sim indicating whether the model is predicting dist or sim
pred_sim_dist = pred_sim_dist

# dist/sim indicating whether the model should supply dist or sim
supply_sim_dist = supply_sim_dist

#######################################
###### ATT+NTN 
#######################################

gcn_num = 3

layer_1 = 'GraphConvolution:output_dim=256,dropout=False,bias=True,act=relu,sparse_inputs=True,type=gcn'
layer_2 = 'GraphConvolution:input_dim=256,output_dim=128,dropout=False,bias=True,act=relu,sparse_inputs=False,type=gcn'
layer_3 = 'GraphConvolution:input_dim=128,output_dim=64,dropout=False,bias=True,act=identity,sparse_inputs=False,type=gcn'
layer_4 = 'JumpingKnowledge:gcn_num=3,gcn_layer_ids=1_2_3,input_dims=256_128_64,att_times=1,att_num=1,att_weight=True,att_style=dot,combine_method=concat'
layer_5 = 'Dense:input_dim=448,output_dim=348,dropout=False,bias=True,act=relu'
layer_6 = 'Dense:input_dim=348,output_dim=256,dropout=False,bias=True,act=relu'
layer_7 = 'Dense:input_dim=256,output_dim=256,dropout=False,bias=True,act=identity'

# Layer index (1-based) to obtain graph embeddings
gemb_layer_id = 7
    
if pred_sim_dist == 'dist':
    layer_8 = 'Dist:norm=None'
else:
    layer_8 = 'Dot:output_dim=1,act=identity'

# number of layers
layer_num = 8
layers = [layer_1, layer_2, layer_3, layer_4, layer_5, layer_6, layer_7, layer_8]

#######################################
###### Graph Loss
#######################################

graph_loss = None

# loss function to use
graph_loss = graph_loss

if graph_loss:
    # Weight parameter for the graph loss function
    graph_loss_alpha = 0.
    
#######################################

train_real_percent = 1

# Use supersource node? It's a Node that is connected to all other nodes in the graph.
supersource = False

########################################
###### Random Walk Generation and Usage
########################################

# Random walk configuration. Set none to not use random walks. Format is: <num_walks>_<walk_length>
random_walk = None

########################################
###### Training (optimization) details.
########################################

# Dropout rate (1 - keep probability)
dropout_rate = 0.

# Weight for L2 loss on embedding matrix
# weight_decay = 0
weight_decay = 0.1

# Initial Learning Rate
learning_rate = 0.001

########################################
###### Training and Validation
########################################

# Which gpu to use, -1 = cpu
# gpu = -1
gpu = 0

# Number of epochs
iters = 500

########################################
###### Testing.
########################################

# Whether to plot the results (involving all baselines) or not
plot_results = True

# Max number of plots per experiment
plot_max_num = 10

# Maximum number of nodes in a graph
max_nodes = max_nodes