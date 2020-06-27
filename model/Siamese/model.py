import config as ec
from layers_factory import create_layers
import numpy as np
import tensorflow as tf
from warnings import warn

logging_enabled = True

class Model(object):
    if logging_enabled == True:
        print("- Entered model::Model Class")
    
    def __init__(self, **kwargs):
        if logging_enabled == True:
            print("- Entered model::Model::__init__ Constructor Method")

        allowed_kwargs = {'name'}
        
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        
        name = kwargs.get('name')
        
        if not name:
            name = self.__class__.__name__.lower()
        
        self.name = name

        self.vars = {}
        self.layers = []
        self.train_loss = 0
        self.val_test_loss = 0
        self.optimizer = None
        self.opt_op = None

        self.batch_size = ec.batch_size
        self.weight_decay = ec.weight_decay
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=ec.learning_rate)

        self._build()
        print('* Flow built')
        
        # Build metrics
        self._loss()
        print('* Loss built')
        
        self.opt_op = self.optimizer.minimize(self.train_loss)
        print('* Optimizer built')

    def _build(self):
        if logging_enabled == True:
            print("- Entered model::Model::_build Private Method")

        # Create layers
        self.layers = create_layers(self, 'layer', ec.layer_num)
        assert (len(self.layers) > 0)
       
        print('\ninfo: {} layers have been Created:\n\t{}\n'.format(
            len(self.layers), ',\n\t'.join(l.get_name() for l in self.layers)))
        
        self.gembd_layer_id = self._gemb_layer_id()
        print('info: index (0-based) of the Graph Embedding Layer: {}'.format(self.gembd_layer_id))

        # Build the siamese model for train and val_test, respectively,
        for tvt in ['train', 'val_test']:                      
            # Go through each layer except the last one.
            acts = [self._get_ins(self.layers[0], tvt)]
            outs = None
            
            if tvt == 'train':
                print('\n* Training:')
            else:
                print('\n* Validation Testing:')

            for k, layer in enumerate(self.layers):
                print("\t", layer.name)
                ins = self._proc_ins(acts[-1], k, layer, tvt)
                outs = layer(ins)
                outs = self._proc_outs(outs, k, layer, tvt)
                acts.append(outs)
            
            if tvt == 'train':
                self.train_outputs = outs
                self.train_acts = acts
                print("train_outputs::outs:\n", self.train_outputs)
                print("train_outputs::acts:\n", self.train_acts)
            else:
                self.val_test_output = outs
                self.val_test_pred_score = self._val_test_pred_score()
                self.val_test_acts = acts
                
                print("val_test_output::outs:\n", self.val_test_output)
                print("val_test_pred_score::acts:\n", self.val_test_pred_score)
                print("val_test_acts::acts:\n", self.val_test_acts)


        self.node_embeddings = self._get_all_gcn_layer_outputs('val_test')

        # Store model variables for easy access.
        variables = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES)
        self.vars = {var.name: var for var in variables}

    def _loss(self):
        if logging_enabled == True:
            print("- Entered model::Model::_loss Private Method")
        
        self.train_loss = self._loss_helper('train')
        self.val_test_loss = self._loss_helper('val')

    def _loss_helper(self, tvt):
        if logging_enabled == True:
            print("- Entered model::Model::_loss_helper Private Method")
        
        rtn = 0

        # Weight decay loss.
        wdl = 0
        for layer in self.layers:
            for var in layer.vars.values():
                wdl = self.weight_decay * tf.nn.l2_loss(var)
                rtn += wdl
        if tvt == 'train':
            tf.compat.v1.summary.scalar('weight_decay_loss', wdl)

        task_loss_dict = self._task_loss(tvt)
        for loss_label, loss in task_loss_dict.items():
            rtn += loss
            if tvt == 'train':
                tf.compat.v1.summary.scalar(loss_label, loss)

        if ec.graph_loss == '1st':
            node_emb_list = self._get_last_gcn_layer_outputs(tvt)
            laplacian_list = self._get_laplacians_for_graph_loss(tvt)
            gl = 0
            
            for i, node_emb_mat in enumerate(node_emb_list):
                # gli = 2 * tf.trace(
                #     dot(tf.transpose(
                #         dot(laplacian_list[i], node_emb_mat, sparse=True)),
                #         node_emb_mat))
                # gl += gli
                mat = tf.matmul(node_emb_mat, tf.transpose(node_emb_mat))
                gl += tf.sqrt(tf.reduce_sum(tf.square(tf.sparse_add(-mat, laplacian_list[i][0]))))
                
            gl /= ec.batch_size
            gl *= ec.graph_loss_alpha
            rtn += gl
            
            if tvt == 'train':
                tf.compat.v1.summary.scalar('1st_order_graph_loss', gl)

        if tvt == 'train':
            tf.compat.v1.summary.scalar('total_loss', rtn)
        
        return rtn

    def _gc_loss_acc(self, logits, labels):
        if logging_enabled == True:
            print("- Entered model::Model::_gc_loss_acc Private Method")
        
        labels = tf.cast(labels, tf.float32)
        logits = tf.cast(logits, tf.float32)

        cross_entropy_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=labels, logits=logits))

        hits = tf.equal(tf.argmax(logits, -1), tf.argmax(labels, -1))
        accuracy = tf.reduce_mean(tf.cast(hits, tf.float32))

        return cross_entropy_loss, accuracy

    def pred_sim_without_act(self):
        raise NotImplementedError()

    def apply_final_act_np(self, score):
        raise NotImplementedError()

    def get_feed_dict_for_train(self, data):
        raise NotImplementedError()

    def get_feed_dict_for_val_test(self, g1, g2, true_sim_dist):
        raise NotImplementedError()

    def get_true_dist_sim(self, i, j, true_result):
        raise NotImplementedError()

    def _get_ins(self, layer, tvt):
        raise NotImplementedError()

    def _proc_ins_for_merging_layer(self, ins, tvt):
        raise NotImplementedError()

    def _val_test_pred_score(self):
        raise NotImplementedError()

    def _task_loss(self, tvt):
        raise NotImplementedError()

    def _proc_ins(self, ins, k, layer, tvt):
        if logging_enabled == True:
            print("- Entered model::Model::_proc_ins Private Method")
        
        ln = layer.__class__.__name__
        ins_mat = None
        
        if k != 0 and tvt == 'train':
            # sparse matrices (k == 0; the first layer) cannot be logged.
            need_log = True
        else:
            need_log = False
        
        if ln == 'GraphConvolution' or ln == 'GraphConvolutionAttention':
            gcn_count = int(layer.name.split('_')[-1])
            assert (gcn_count >= 1)  # 1-based
            gcn_id = gcn_count - 1
            ins = self._supply_laplacians_etc_to_ins(ins, tvt, gcn_id)
            if need_log:
                ins_mat = self._stack_concat([i[0] for i in ins])
        
        # For Multi-Level GCN
        elif ln == 'GraphConvolutionCollector' or ln == 'JumpingKnowledge':
            ins = []
            for lr in self.layers:
                if lr.__class__.__name__ == 'GraphConvolution':
                    ins.append(lr.output)
            ins_mat = tf.constant([])
        
        else:
            ins_mat = self._stack_concat(ins)
            if layer.merge_graph_level_embs():
                ins = self._proc_ins_for_merging_layer(ins, tvt)
            if ln == 'Dense' and self._has_seen_merge_layer(k):
                # Use matrix operations instead of iterating through list
                # after the merging layer.
                ins = ins_mat
        
        if need_log:
            self._log_mat(ins_mat, layer, 'ins')
        
        return ins

    def _proc_outs(self, outs, k, layer, tvt):
        if logging_enabled == True:
            print("- Entered model::Model::_proc_outs Private Method")
        
        outs_mat = self._stack_concat(outs)
        ln = layer.__class__.__name__
        
        if tvt == 'train':
            self._log_mat(outs_mat, layer, 'outs')
        
        if k == self.gembd_layer_id:
            if ln != 'ANPM' and ln != 'ANPMD' and ln != 'ANNH':
                embs = outs
            else:
                embs = layer.embeddings
            assert (type(embs) is list)
            
            # Note: some architecture may NOT produce
            # any graph-level embeddings.
            if tvt == 'train':
                self.graph_embeddings_train = embs
            elif tvt == 'val_test':
                self.graph_embeddings_val_test = embs  # for train.py to collect
                s = embs[0].get_shape().as_list()
                assert (s[0] == 1)
                self.gemb_dim = s[1]  # for train.py to collect
            else:
                assert (False)
        
        if tvt == 'val_test' and layer.produce_node_atts():
            if ln == 'Attention':
                assert (len(outs) == 2)
            self.attentions = layer.att
            s = self.attentions.get_shape().as_list()
            assert (s[1] == 1)
        
        return outs

    def _supply_laplacians_etc_to_ins(self, ins, tvt, gcn_id):
        if logging_enabled == True:
            print("- Entered model::Model::_supply_laplacians_etc_to_ins Private Method")
        
        rtn = []
        
        if not ec.coarsening:
            gcn_id = 0
        
        for i, (laplacians, num_nonzero, edge_index, incidence_mat) in \
                enumerate(zip(
                    self._get_plhdr('laplacians_1', tvt) +
                    self._get_plhdr('laplacians_2', tvt),
                    self._get_plhdr('num_nonzero_1', tvt) +
                    self._get_plhdr('num_nonzero_2', tvt),
                    self._get_plhdr('edge_index_1', tvt) +
                    self._get_plhdr('edge_index_2', tvt),
                    self._get_plhdr('incidence_mat_1', tvt) +
                    self._get_plhdr('incidence_mat_2', tvt)
                )):
            
            print(
                "ins[i]: ", ins[i], 
                ", laplacians[gcn_id]: ", laplacians[gcn_id], 
                ", num_nonzero: ", num_nonzero, 
                ", edge_index: ", edge_index, 
                ", incidence_mat: ", incidence_mat) 
            
            rtn.append([ins[i], laplacians[gcn_id], num_nonzero, edge_index, incidence_mat])
        
        return rtn

    def _has_seen_merge_layer(self, k):
        if logging_enabled == True:
            print("- Entered model::Model::_has_seen_merge_layer Private Method")
        
        for i, layer in enumerate(self.layers):
            if i < k and layer.merge_graph_level_embs():
                return True
        
        return False

    def _gemb_layer_id(self):
        if logging_enabled == True:
            print("- Entered model::Model::_gemb_layer_id Private Method")
        
        id = ec.gemb_layer_id
        
        if id is not None:
            assert (id >= 1)
            id -= 1
        
        else:
            for i, layer in enumerate(self.layers):
                if layer.produce_graph_level_emb() and ec.coarsening:
                    return i
        
        return id

    def _get_plhdr(self, key, tvt):
        if logging_enabled == True:
            print("- Entered model::Model::_get_plhdr Private Method")
        
        if tvt == 'train':
            return self.__dict__[key]
        else:
            assert (tvt == 'test' or tvt == 'val' or tvt == 'val_test')
            return self.__dict__['val_test_' + key]

    def _get_last_gcn_layer_outputs(self, tvt):
        if logging_enabled == True:
            print("- Entered model::Model::_get_last_gcn_layer_outputs Private Method")
        
        return self._get_all_gcn_layer_outputs(tvt)[-1]

    def _get_all_gcn_layer_outputs(self, tvt):
        if logging_enabled == True:
            print("- Entered model::Model::_get_all_gcn_layer_outputs Private Method")
        
        rtn = []
        acts = self.train_acts if tvt == 'train' else self.val_test_acts
        assert (len(acts) == len(self.layers) + 1)
        
        for k, layer in enumerate(self.layers):
            ln = layer.__class__.__name__
            if ln == 'GraphConvolution' or ln == 'GraphConvolutionAttention':
                rtn.append(acts[k + 1])  # the 0th is the input
        
        return rtn

    def _get_laplacians_for_graph_loss(self, tvt):
        if logging_enabled == True:
            print("- Entered model::Model::_get_laplacians_for_graph_loss Private Method")
        
        rtn = []
        
        for laplacians in (self._get_plhdr('laplacians_1', tvt) +
                           self._get_plhdr('laplacians_2', tvt)):
            assert (len(laplacians) == 1)
            rtn.append(laplacians[0])
        
        return rtn

    def _get_output_of_a_specific_layer(self, layer_name, tvt):
        raise NotImplementedError()  # TODO: for Yunsheng Bai

    def _stack_concat(self, x):
        if logging_enabled == True:
            print("- Entered model::Model::_stack_concat Private Method")
        
        if type(x) is list:
            list_of_tensors = x
            assert (list_of_tensors)
            s = list_of_tensors[0].get_shape()
            if s != ():
                return tf.concat(list_of_tensors, 0)
            else:
                return tf.stack(list_of_tensors)
        else:
            # assert(len(x.get_shape()) == 2) # should be a 2-D matrix
            return x

    def _log_mat(self, mat, layer, label):
        if logging_enabled == True:
            print("- Entered model::Model::_log_mat Private Method")
        
        tf.compat.v1.summary.histogram(layer.name + '_' + label, mat)

    def save(self, sess, saver, iter):
        if logging_enabled == True:
            print("- Entered model::Model::save Public Method")
        
        logdir = saver.get_log_dir()
        sp = '{}\\models\\{}.ckpt'.format(logdir, iter)
        tf.train.Saver(self.vars).save(sess, sp)

    def load(self, sess, load_path):
        if logging_enabled == True:
            print("- Entered model::Model::load Public Method")
        
        tf.train.Saver(self.vars).restore(sess, load_path)
