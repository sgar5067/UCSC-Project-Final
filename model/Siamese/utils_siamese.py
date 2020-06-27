import config as ec
import sys
from os.path import dirname, abspath, join
import tensorflow as tf

cur_folder = dirname(abspath(__file__))
sys.path.insert(0, '{}\\..\\..\\src'.format(cur_folder))
from utils import sorted_nicely, get_ts


def solve_parent_dir():
    pass

def extract_config_code():
    with open(join(get_siamese_dir(), 'config.py')) as f:
        return f.read()

def convert_msec_to_sec_str(sec):
    return '{:.2f}msec'.format(sec * 1000)

def convert_long_time_to_str(sec):
    day = sec // (24 * 3600)
    sec = sec % (24 * 3600)
    hour = sec // 3600
    sec %= 3600
    minutes = sec // 60
    sec %= 60
    seconds = sec
    return '{} days {} hours {} mins {:.1f} secs'.format(
        int(day), int(hour), int(minutes), seconds)

def get_siamese_dir():
    return cur_folder

def get_coarsen_level():
    if ec.coarsening:
        return ec.coarsening[6:]
    else:
        return 1

def get_model_info_as_str(model_info_table=None):
    rtn = []
    d = FLAGS.flag_values_dict()
    for k in sorted_nicely(d.keys()):
        v = d[k]
        s = '{0:26} : {1}'.format(k, v)
        rtn.append(s)
        if model_info_table:
            model_info_table.append([k, '**{}**'.format(v)])
    rtn.append('{0:26} : {1}'.format('ts', get_ts()))
    return '\n'.join(rtn)

def dot(x, y, sparse=False):
    # Wrapper for tf.matmul (sparse vs dense)
    if sparse:
        res = tf.sparse.sparse_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res

def truth_is_dist_sim():
    if ds_metric == 'ged':
        sim_or_dist = 'dist'
    else:
        sim_or_dist = 'sim'
    return sim_or_dist