from utils import format_float
import numpy as np

logging_enabled = True

class SimilarityKernel(object):
    if logging_enabled == True:
        print("- Entered dist_sim_kernel::SimilarityKernel Class")

    def name(self):
        if logging_enabled == True:
            print("- Entered dist_sim_kernel::SimilarityKernel::name Public Method")

        return self.__class__.__name__.lower()

    def shortname(self):
        if logging_enabled == True:
            print("- Entered dist_sim_kernel::SimilarityKernel::shortname Public Method")

        return self.name()

    def name_suffix(self):
        if logging_enabled == True:
            print("- Entered dist_sim_kernel::SimilarityKernel::name_suffix Public Method")

        return ''

    def dist_to_sim_np(self, dist, max_dist):
        raise NotImplementedError()

    def dist_to_sim_tf(self, dist, max_dist):
        raise NotImplementedError()

    def sim_to_dist_np(self, sim):
        raise NotImplementedError()


# class IdentityKernel:
#     def dist_to_sim_np(self, dist, max_dist):
#         return self._d_to_s(dist, max_dist)
#
#     def dist_to_sim_tf(self, dist, max_dist):
#         return self._d_to_s(dist, max_dist)
#
#     def _d_to_s(self, dist, max_dist):
#         return 1 - dist / max_dist

class IdentityKernel:
    if logging_enabled == True:
        print("- Entered dist_sim_kernel::IdentityKernel Class")

    def dist_to_sim_np(self, dist, *unused):
        if logging_enabled == True:
            print("- Entered dist_sim_kernel::IdentityKernel::dist_to_sim_np Public Method")

        return self._d_to_s(dist)

    def dist_to_sim_tf(self, dist, *unused):
        if logging_enabled == True:
            print("- Entered dist_sim_kernel::IdentityKernel::dist_to_sim_tf Public Method")

        return self._d_to_s(dist)

    def sim_to_dist_np(self, sim):
        if logging_enabled == True:
            print("- Entered dist_sim_kernel::IdentityKernel::sim_to_dist_np Public Method")

        return sim

    def _d_to_s(self, dist):
        if logging_enabled == True:
            print("- Entered dist_sim_kernel::IdentityKernel::_d_to_s Private Method")

        return dist


class GaussianKernel(SimilarityKernel):
    if logging_enabled == True:
        print("- Entered dist_sim_kernel::GaussianKernel Class")

    def __init__(self, yeta):
        if logging_enabled == True:
            print("- Entered dist_sim_kernel::GaussianKernel::__init__ Constructor Method")

        self.yeta = yeta

    def name(self):
        if logging_enabled == True:
            print("- Entered dist_sim_kernel::GaussianKernel::name Public Method")

        return 'Gaussian_yeta={}'.format(format_float(self.yeta))

    def shortname(self):
        if logging_enabled == True:
            print("- Entered dist_sim_kernel::GaussianKernel::shortname Public Method")

        return 'g_{:.2e}'.format(self.yeta)

    def dist_to_sim_np(self, dist, *unused):
        if logging_enabled == True:
            print("- Entered dist_sim_kernel::GaussianKernel::dist_to_sim_np Public Method")

        return np.exp(-self.yeta * np.square(dist))

    def dist_to_sim_tf(self, dist, *unuse):
        if logging_enabled == True:
            print("- Entered dist_sim_kernel::GaussianKernel::dist_to_sim_tf Public Method")

        import tensorflow as tf
        return tf.exp(-self.yeta * tf.square(dist))

    def sim_to_dist_np(self, sim):
        if logging_enabled == True:
            print("- Entered dist_sim_kernel::GaussianKernel::sim_to_dist_np Public Method")

        return np.sqrt(-np.log(sim) / self.yeta)


class ExpKernel(SimilarityKernel):
    if logging_enabled == True:
        print("- Entered dist_sim_kernel::ExpKernel Class")

    def __init__(self, scale):
        if logging_enabled == True:
            print("- Entered dist_sim_kernel::ExpKernel::__init__ Constructor Method")

        self.scale = scale

    def name(self):
        if logging_enabled == True:
            print("- Entered dist_sim_kernel::ExpKernel::name Public Method")

        return 'Exp_scale={}'.format(format_float(self.scale))

    def shortname(self):
        if logging_enabled == True:
            print("- Entered dist_sim_kernel::ExpKernel::shortname Public Method")

        return 'e_{:.2e}'.format(self.scale)

    def dist_to_sim_np(self, dist, *unused):
        # if logging_enabled == True:
        #     comment out unless needed. The method is called to often
        #     print("- Entered dist_sim_kernel::ExpKernel::dist_to_sim_np Public Method")

        return np.exp(-self.scale * dist)

    def dist_to_sim_tf(self, dist, *unuse):
        if logging_enabled == True:
            print("- Entered dist_sim_kernel::ExpKernel::dist_to_sim_tf Public Method")

        import tensorflow as tf
        return tf.exp(-self.scale * dist)

    def sim_to_dist_np(self, sim):
        if logging_enabled == True:
            print("- Entered dist_sim_kernel::ExpKernel::sim_to_dist_np Public Method")

        # if sim == 0.0:
        #     # e.g. if dist = 748, then sim = 0.0, and log(0.0) leads to a warning
        #     rtn = np.inf
        # else:
        rtn = -np.log(sim) / self.scale
        return rtn


class InverseKernel(SimilarityKernel):
    if logging_enabled == True:
        print("- Entered dist_sim_kernel::InverseKernel Class")

    def __init__(self, scale):
        if logging_enabled == True:
            print("- Entered dist_sim_kernel::InverseKernel::__init__ Constructor Method")

        self.scale = scale

    def name(self):
        if logging_enabled == True:
            print("- Entered dist_sim_kernel::InverseKernel::name Public Method")

        return 'Inverse_scale={}'.format(format_float(self.scale))

    def shortname(self):
        if logging_enabled == True:
            print("- Entered dist_sim_kernel::InverseKernel::logging_enabled Public Method")

        return 'i_{:.2e}'.format(self.scale)

    def dist_to_sim_np(self, dist, *unused):
        if logging_enabled == True:
            print("- Entered dist_sim_kernel::InverseKernel::dist_to_sim_np Public Method")

        return 1 / (self.scale * dist + 1)

    def dist_to_sim_tf(self, dist, *unuse):
        if logging_enabled == True:
            print("- Entered dist_sim_kernel::InverseKernel::dist_to_sim_tf Public Method")

        return 1 / (self.scale * dist + 1)

    def sim_to_dist_np(self, sim):
        if logging_enabled == True:
            print("- Entered dist_sim_kernel::InverseKernel::sim_to_dist_np Public Method")

        return (1 / sim - 1) / self.scale


def create_ds_kernel(kernel_name, yeta=None, scale=None):
    if logging_enabled == True:
        print("- Entered dist_sim_kernel::create_ds_kernel Global Method")

    if kernel_name == 'identity':
        return IdentityKernel()
    elif kernel_name == 'gaussian':
        return GaussianKernel(yeta)
    elif kernel_name == 'exp':
        return ExpKernel(scale)
    elif kernel_name == 'inverse':
        return InverseKernel(scale)
    else:
        raise RuntimeError('Unknown sim kernel {}'.format(kernel_name))
