from utils import get_train_str, get_data_path, get_save_path, sorted_nicely, \
    save, load, append_ext_to_filepath
import networkx as nx
import numpy as np
import random
from random import randint
from collections import OrderedDict
from glob import glob
from os.path import basename, join

logging_enabled = True

class Data(object):
    if logging_enabled == True:
        print("- Entered data::Data Class")

    def __init__(self, name_str):
        if logging_enabled == True:
            print("- Entered data::Data::__init__ Constructor Method")

        name = join(self.__class__.__name__, name_str + self.name_suffix())
        sfn = self.save_filename(name)
        temp = load(sfn, use_klepto=True)  # use klepto for faster saving and loading
        if temp:
            self.__dict__ = temp
            print('info: {}\nData config obtained from {}{}\n'.format(
                name, sfn,
                ' with {} graphs'.format(
                    len(self.graphs)) if
                hasattr(self, 'graphs') else ''))
        else:
            self.init()
            save(sfn, self.__dict__)
            print('info: {} saved to {}\n'.format(name, sfn))

    def init(self):
        raise NotImplementedError()

    def name_suffix(self):
        if logging_enabled == True:
            print("- Entered data::Data::name_suffix Public Method")
        
        return ''

    def save_filename(self, name):
        if logging_enabled == True:
            print("- Entered data::Data::save_filename Public Method")
        
        return '{}\\{}'.format(get_save_path(), name)

    def get_gids(self):
        if logging_enabled == True:
            print("- Entered data::Data::get_gids Public Method")
        
        return [g.graph['gid'] for g in self.graphs]


class SynData(Data):
    if logging_enabled == True:
        print("- Entered data::SynData Class")

    train_num_graphs = 20
    test_num_graphs = 10

    def __init__(self, train):
        if train:
            self.num_graphs = SynData.train_num_graphs
        else:
            self.num_graphs = SynData.test_num_graphs
        super(SynData, self).__init__(get_train_str(train))

    def init(self):
        self.graphs = []
        for i in range(self.num_graphs):
            n = randint(5, 20)
            m = randint(n - 1, n * (n - 1) / 2)
            g = nx.gnm_random_graph(n, m)
            g.graph['gid'] = i
            self.graphs.append(g)
        print('info: Randomly generated %s graphs' % self.num_graphs)

    def name_suffix(self):
        return '_{}_{}'.format(SynData.train_num_graphs,
                               SynData.test_num_graphs)


class IMDBMultiData(Data):
    if logging_enabled == True:
        print("- Entered data::IMDBMultiData Class")

    def __init__(self, train):
        self.train = train
        super(IMDBMultiData, self).__init__(get_train_str(train))

    def init(self):
        dir = 'IMDBMulti'
        self.graphs = get_proc_graphs(dir, self.train)
        self.graphs, self.glabels = add_glabel_to_each_graph(self.graphs, dir)


class NCI109Data(Data):
    if logging_enabled == True:
        print("- Entered data::NCI109Data Class")

    def __init__(self, train):
        self.train = train
        super(NCI109Data, self).__init__(get_train_str(train))

    def init(self):
        dir = 'NCI109'
        self.graphs = get_proc_graphs(dir, self.train)
        self.graphs, self.glabels = add_glabel_to_each_graph(self.graphs, dir)


class WebEasyData(Data):
    if logging_enabled == True:
        print("- Entered data::WebEasyData Class")

    def __init__(self, train):
        self.train = train
        super(WebEasyData, self).__init__(get_train_str(train))

    def init(self):
        dir = 'WEBEASY'
        self.graphs = get_proc_graphs(dir, self.train)
        self.graphs, self.glabels = add_glabel_to_each_graph(self.graphs, dir)
        assert (self.glabels is not None)  # real graph labels


class Reddit5kData(Data):
    if logging_enabled == True:
        print("- Entered data::Reddit5kData Class")

    def __init__(self, train):
        self.train = train
        super(Reddit5kData, self).__init__(get_train_str(train))

    def init(self):
        dir = 'RedditMulti5k'
        self.graphs = get_proc_graphs(dir, self.train)
        self.graphs, self.glabels = add_glabel_to_each_graph(self.graphs, dir)
        assert (self.glabels is not None)  # real graph labels


class Reddit10kData(Data):
    if logging_enabled == True:
        print("- Entered data::Reddit10kData Class")

    def __init__(self, train):
        self.train = train
        super(Reddit10kData, self).__init__(get_train_str(train))

    def init(self):
        dir = 'RedditMulti10k'
        self.graphs = get_proc_graphs(dir, self.train)
        self.graphs, self.glabels = add_glabel_to_each_graph(self.graphs, dir)
        assert (self.glabels is not None)  # real graph labels
        self.further_sample()

    def further_sample(self):
        return


class Reddit10kSmallData(Reddit10kData):
    if logging_enabled == True:
        print("- Entered data::Reddit10kSmallData Class")

    def further_sample(self):
        self.graphs_small = []
        self.glabels_small = []
        self.ids = []
        for i, g in enumerate(self.graphs):
            if g.number_of_nodes() < 25:
                self.graphs_small.append(g)
                self.glabels_small.append(self.glabels[i])
                self.ids.append(i)
        self.graphs = self.graphs_small
        self.glabels = self.glabels_small
        print('info: Sampled {} small graphs'.format(len(self.graphs)))


class PTCData(Data):
    if logging_enabled == True:
        print("- Entered data::PTCData Class")

    def __init__(self, train):
        self.train = train
        super(PTCData, self).__init__(get_train_str(train))

    def init(self):
        dir = 'PTC'
        self.graphs = get_proc_graphs(dir, self.train)
        self.graphs, self.glabels = add_glabel_to_each_graph(self.graphs, dir)
        assert (self.glabels is not None)  # real graph labels


class ProjectData(Data):
    if logging_enabled == True:
        print("- Entered data::ProjectData Class")

    def __init__(self, train):
        self.train = train
        super(ProjectData, self).__init__(get_train_str(train))

    def init(self):
        dir = 'Project'
        self.graphs = get_proc_graphs(dir, self.train)
        self.graphs, self.glabels = add_glabel_to_each_graph(self.graphs, dir)
        assert (self.glabels is not None)  # real graph labels
        
def get_proc_graphs(datadir, train):
    if logging_enabled == True:
        print("- Entered data::get_proc_graphs Global Method")

    datadir = '{}\\{}\\{}'.format(
        get_data_path(), datadir, get_train_str(train))
    graphs = iterate_get_graphs(datadir)
    print('info: Loaded {} graphs from {}'.format(len(graphs), datadir))
    return graphs


def iterate_get_graphs(dir):
    if logging_enabled == True:
        print("- Entered data::iterate_get_graphs Global Method")

    graphs = []
    for file in sorted_nicely(glob(dir + '\\*.gexf')):
        gid = int(basename(file).split('.')[0])
        g = nx.read_gexf(file)
        g.graph['gid'] = gid                      
        graphs.append(g)
        
        if not nx.is_connected(g):
            print('info: {} not connected'.format(gid))
    return graphs


""" Graph labels. """

def add_glabel_to_each_graph(graphs, dir, use_fake_glabels=False):
    if logging_enabled == True:
        print("- Entered data::add_glabel_to_each_graph Global Method")

    glabels = None
    if not use_fake_glabels:
        filepath = '{}\\{}\\glabels.txt'.format(get_data_path(), dir)
        glabels = load_glabels_from_txt(filepath)        
    seen = set()  # check every graph id is seen only once
    for g in graphs:
        gid = g.graph['gid']
        assert (gid not in seen)
        seen.add(gid)
        if use_fake_glabels:
            glabel = randint(0, 9)  # randomly assign a graph label from {0, .., 9}
        else:
            glabel = glabels[gid]
        g.graph['glabel'] = glabel
    return graphs, glabels


def save_glabels_as_txt(filepath, glabels):
    if logging_enabled == True:
        print("- Entered data::save_glabels_as_txt Global Method")

    filepath = append_ext_to_filepath('.txt', filepath)
    with open(filepath, 'w') as f:
        for id, glabel in OrderedDict(glabels).items():
            f.write('{}\t{}\n'.format(id, glabel))


def load_glabels_from_txt(filepath):
    if logging_enabled == True:
        print("- Entered data::load_glabels_from_txt Global Method")

    filepath = append_ext_to_filepath('.txt', filepath)
    rtn = {}
    int_map = {}
    seen_glabels = set()
    with open(filepath) as f:
        for line in f:
            ls = line.rstrip().split()
            assert (len(ls) == 2)
            gid = int(ls[0])
            try:
                glabel = int(ls[1])
            except ValueError:
                label_string = ls[1]
                glabel = int_map.get(label_string)
                if glabel is None:
                    glabel = len(int_map)  # guarantee 0-based
                    int_map[label_string] = glabel  # increase the size of int_map by 1
            rtn[gid] = glabel
            seen_glabels.add(glabel)
    if 0 not in seen_glabels:  # check 0-based graph labels
        raise RuntimeError('{} has no glabel 0; {}'.format(filepath, seen_glabels))
    return rtn