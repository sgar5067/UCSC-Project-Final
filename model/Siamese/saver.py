import config as ec

from utils import get_ts, create_dir_if_not_exists, save_as_dict
from utils_siamese import get_siamese_dir
import tensorflow as tf
from glob import glob
from os import system
from collections import OrderedDict
from pprint import pprint
from os.path import join

logging_enabled = True

class Saver(object):
    if logging_enabled == True:
        print("- Entered saver::Saver Class")
    
    def __init__(self, sess=None):
        if logging_enabled == True:
            print("- Entered saver::Saver::__init__ Constructor Method")

        model_str = self._get_model_str()
        self.logdir = '{}\\logs\\{}'.format(get_siamese_dir(), model_str)
        create_dir_if_not_exists(self.logdir)
        
        if sess is not None:
            self.tw = tf.compat.v1.summary.FileWriter(self.logdir + '\\train', sess.graph)
            self.all_merged = tf.compat.v1.summary.merge_all()
            self.loss_merged = tf.compat.v1.summary.merge(self._extract_loss_related_summaries_as_list())
        
        self.f = open('{}\\results.txt'.format(self.logdir), 'w')
        print('Logging to {}'.format(self.logdir))

    def get_log_dir(self):
        if logging_enabled == True:
            print("- Entered saver::Saver::get_log_dir Public Method")

        return self.logdir

    def proc_objs(self, objs, tvt, iter):
        if logging_enabled == True:
            print("- Entered saver::Saver::proc_objs Public Method")

        if 'train' in tvt:
            objs.insert(0, self.loss_merged)
        
        return objs

    def proc_outs(self, outs, tvt, iter):
        if logging_enabled == True:
            print("- Entered saver::Saver::proc_outs Public Method")

        if 'train' in tvt:
            # print("outs[0]: ", outs[0], ", iter: ", iter)
            self.tw.add_summary(outs[0], iter)

    def save_test_info(self, node_embs_dict, graph_embs_mat, emb_time):
        if logging_enabled == True:
            print("- Entered saver::Saver::save_test_info Public Method")

        sfn = '{}\\test_info'.format(self.logdir)
        
        # The following function call must be made in one line!
        save_as_dict(sfn, node_embs_dict, graph_embs_mat, emb_time)

    def save_conf_code(self, conf_code):
        if logging_enabled == True:
            print("- Entered saver::Saver::save_conf_code Public Method")

        with open(join(self.logdir, 'config.py'), 'w') as f:
            f.write(conf_code)

    def save_overall_time(self, overall_time):
        if logging_enabled == True:
            print("- Entered saver::Saver::save_overall_time Public Method")

        self._save_to_result_file(overall_time, 'overall time')

    def clean_up_saved_models(self, best_iter):
        if logging_enabled == True:
            print("- Entered saver::Saver::clean_up_saved_models Public Method")

        for file in glob('{}\\models\\*'.format(self.get_log_dir())):
            
            if str(best_iter) not in file:
                system('rm -rf {}'.format(file))

    def _get_model_str(self):
        if logging_enabled == True:
            print("- Entered saver::Saver::_get_model_str Private Method")

        li = []
        key_flags = [ec.model, ec.dataset_train]
        if ec.dataset_val_test != ec.dataset_train:
            key_flags.append(ec.dataset_val_test)
        
        for f in key_flags:
            li.append(str(f))
        
        return '_'.join(li)

    def _log_model_info(self, logdir, sess):
        if logging_enabled == True:
            print("- Entered saver::Saver::_log_model_info Private Method")

        model_info_table = [["**key**", "**value**"]]
        
        with open(logdir + '\\model_info.txt', 'w') as f:
            s = get_model_info_as_str(model_info_table)
            f.write(s)
        
        model_info_op = tf.compat.v1.summary.text('model_info', tf.convert_to_tensor(model_info_table))
        
        if sess is not None:
            print("model_info_op: ", model_info_op)
            self.tw.add_summary(sess.run(model_info_op))

    def _save_to_result_file(self, obj, name):
        if logging_enabled == True:
            print("- Entered saver::Saver::_save_to_result_file Private Method")

        if type(obj) is dict or type(obj) is OrderedDict:
            # print("obj: ", obj, ", self.f: ", self.f)
            pprint(obj, stream=self.f)
            
        else:
            # print("obj: ", obj, ", self.f: ", self.f, ", name: ", name)
            self.f.write('{}: {}\n'.format(name, obj))

    def _extract_loss_related_summaries_as_list(self):
        if logging_enabled == True:
            print("- Entered saver::Saver::_extract_loss_related_summaries_as_list Private Method")

        rtn = []
        for tensor in tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.SUMMARIES):
            
            # print("tensor.name: ", tensor.name)
            # Assume "loss" is in the loss-related summary tensors.
            if 'loss' in tensor.name:
                rtn.append([tensor])
        
        return rtn

    def _bool_to_str(self, b, s):
        if logging_enabled == True:
            print("- Entered saver::Saver::_bool_to_str Private Method")

        assert (type(b) is bool)
        
        if b:
            return s
        else:
            return 'NO{}'.format(s)
