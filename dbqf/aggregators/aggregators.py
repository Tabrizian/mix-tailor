import torch
import bisect
import torch
import numpy as np

from dbqf.utils import flatten_grad, unflatten


class Aggregator(object):

    def __init__(self, logger, optimizer, world_size, num_byz=0, attack=None,
                 grad_hook=None):
        self.world_size = world_size
        self.logger = logger
        self.optimizer = optimizer
        self.num_byz = num_byz
        self.grad_hook = grad_hook

    def set_grads(self, grads):
        self.grads = grads

    def agg_grad(self, grads):
        agg_grad_res = self.agg_g(grads)
        return agg_grad_res

    def agg_grad_byz(self, grads):
        # aggregate gradients when there are byzantine workers it requires two
        # passes compared to non-byzantine aggregation which only requires a
        # single pass
        optimizer = self.optimizer
        num_byz = self.num_byz

        self.logger.log_value('byz_grads', self.num_byz, optimizer.niters)
        self.logger.log_value('total_grads', len(grads), optimizer.niters)
        # number of non-byzantine
        attack_grads = []
        num_nonbyz = self.world_size - self.num_byz
        for i in range(self.num_byz):
            attack_grad = self.attack.grad(grads[0:num_nonbyz])
            attack_grads.append(attack_grad)

        grads[-num_byz:] = torch.stack(attack_grads)

        grad = self.agg_g(grads)
        return grad

    def agg_grad_nobyz(self, grads):
        grad = self.agg_g(grads)
        return grad

    def find_closest_grad(self, grads, grad, mask):
        # returns the index of the closest gradient to grad in grads
        distances = torch.norm(grads - grad, dim=1)
        return torch.argsort(distances)[mask][0]

    def find_grad(self, grads, grad):
        i = 0
        for i in range(len(grads)):
            if (grads[i] == grad).all():
                return i

        return None

    def agg_g(self, grads):
        # core function to be overrided by subclasses
        raise NotImplementedError

    def agg_weight(self, model):
        raise NotImplemented


class FedAvg(Aggregator):

    def agg_weight(self, model):
        size = self.world_size

        with torch.no_grad():
            for param in model.parameters():
                dist.all_reduce(param, op=dist.ReduceOp.SUM)
                param /= size

    def agg_g(self, grads):
        return torch.mean(grads, dim=0)

    def __str__(self):
        return f"<FedAvg>"


class Krum(Aggregator):
    # krum calculates score for each of the gradients
    # implementation of https://arxiv.org/abs/1905.04374

    def __init__(self, logger, optimizer, world_size, num_byz=0, attack=None,
                 grad_hook=None, norm='fro', num_krum=1):
        super().__init__(logger, optimizer, world_size, num_byz, attack, grad_hook)
        self.num_byz = num_byz
        self.attack = attack
        self.norm = norm
        self.num_krum = num_krum

    def agg_g(self, grads):
        num_byz = self.num_byz
        logger = self.logger
        optimizer = self.optimizer
        norm = self.norm
        num_krum = self.num_krum

        num_near = len(grads) - num_byz - 2

        # make sure there are enough gradients
        assert num_near > 0

        scores = []
        for grad in grads:
            dists = torch.norm(grads - grad, norm, dim=1)
            dists, _ = torch.sort(dists)
            dists = dists[1:]
            scores.append(torch.sum(dists[:num_near]))

        selected_grads = torch.argsort(torch.tensor(scores))[:num_krum]

        if logger:
            for i in range(len(selected_grads)):
                logger.log_value('krum/select', selected_grads[i],
                                 optimizer.niters)

        if num_krum == 1:
            return grads[selected_grads[0]]
        else:
            return torch.stack([
                grads[selected_grads[i]] for i in range(num_krum)
            ])

    def __str__(self):
        return f"<Krum Aggregator with norm={self.norm}>"


class GeoMed(Aggregator):

    def __init__(self, logger,  optimizer, world_size, num_byz=0, attack=None,
                 grad_hook=None, norm='fro', iters=20):
        # the algorithm currently only works for euclicidian norm
        super().__init__(logger, optimizer, world_size, num_byz, attack, grad_hook)
        self.norm = norm
        self.iters = iters

    def agg_g(self, grads):
        # calculation is based on algorithm below
        # https://en.wikipedia.org/wiki/Geometric_median#Computation
        iters = self.iters

        y = torch.rand(len(grads[0])).to(grads[0].device)
        for i in range(iters):
            num = 0
            for j in range(len(grads)):
                num += grads[j] / torch.norm(grads[j] - y)
            denum = 0
            for j in range(len(grads)):
                denum += 1 / torch.norm(grads[j] - y)
            y = num / denum
        return y

    def __str__(self):
        return f"<Geomed Aggregator with norm={self.norm}>"


class TrimmedMean(Aggregator):

    def __init__(self, logger,  optimizer, world_size, num_byz=0, attack=None):
        # the algorithm currently only works for euclicidian norm
        super().__init__(logger, optimizer, world_size, num_byz, attack)
        self.norm = norm
        self.iters = iters


class CoMed(Aggregator):

    def __init__(self, logger,  optimizer, world_size, num_byz=0, attack=None, beta=1):
        # coordinate-wise median algorithm as described in bulyan paper
        super().__init__(logger, optimizer, world_size, num_byz, attack)

        if beta is None:
            n = self.world_size
            f = self.num_byz
            theta = n - 2 * f
            self.beta = theta - 2 * f
        else:
            self.beta = beta

    def agg_g(self, grads):
        beta = self.beta

        # finding beta closest coordinates to the median you can sort the each
        # coordinate and find the beta closest gradients
        dist_median = torch.abs(grads -
                                torch.median(grads, dim=0)[0])

        # select indexes
        idxs = torch.sort(dist_median, dim=0)[1]
        selection_set = torch.gather(grads, 0, idxs)
        return torch.mean(selection_set[:beta, :], dim=0)

    def __str__(self):
        return f"<Comed Aggregator with beta={self.beta}>"


class Brute(Aggregator):

    def __init__(self, logger, optimizer, world_size, num_byz=0, attack=None,
                 grad_hook=None):
        super().__init__(logger, optimizer, num_byz=num_byz,
                         attack=attack, grad_hook=grad_hook)


class Bulyan(Aggregator):
    # implementation of Bulyan aggregator according to
    # https://arxiv.org/pdf/1802.07927.pdf
    # The Hidden Vulnerability of Distributed Learning in Byzantium
    # Default aggregator used here is Krum

    def __init__(self, logger, optimizer, world_size, num_byz=0, attack=None,
                 grad_hook=None, sel_agg=None, agg_agg=None):
        super().__init__(logger, optimizer, world_size, num_byz=num_byz,
                         attack=attack, grad_hook=grad_hook)

        # selection phase aggregator
        self.sel_agg = sel_agg
        # aggregation phase aggregator
        self.agg_agg = agg_agg

    def agg_g(self, grads):
        # aggregation rule implementation according to section 4 of the
        # paper

        n = self.world_size
        f = self.num_byz
        theta = n - 2 * f

        assert n - 4 * f - 3 >= 0

        logger = self.logger
        optimizer = self.optimizer

        selection_set = []
        mask = torch.arange(len(grads)) != -1

        # selection phase
        for i in range(theta):
            aggregated_grad = self.sel_agg.agg_g(grads[mask])
            closest_idx = self.find_closest_grad(grads, aggregated_grad, mask)
            # index of client
            if logger:
                logger.log_value('bulyan/select', closest_idx, optimizer.niters)
            selection_set.append(grads[closest_idx])

            # removing the closest vector
            mask[closest_idx] = False

        # aggregation phase
        beta = theta - 2 * f
        selection_set = torch.stack(selection_set)

        return self.agg_agg.agg_g(selection_set)

    def __str__(self):
        return f"<Bulyan with selector aggregator {self.sel_agg} final aggregator {self.agg_agg}>"


class BulyanP1(Aggregator):
    # Same as bulyan but without phase 2

    def __init__(self, logger, optimizer, world_size, num_byz=0, attack=None,
                 grad_hook=None, sel_agg=None, agg_agg=None):
        super().__init__(logger, optimizer, world_size, num_byz=num_byz,
                         attack=attack, grad_hook=grad_hook)

        # selection phase aggregator
        self.sel_agg = sel_agg
        # aggregation phase aggregator
        self.agg_agg = agg_agg

    def agg_g(self, grads):
        # aggregation rule implementation according to section 4 of the
        # paper

        n = self.world_size
        f = self.num_byz
        theta = n - 2 * f

        logger = self.logger
        optimizer = self.optimizer

        assert n - 4 * f - 3 >= 0

        selection_set = []
        mask = torch.arange(len(grads)) != -1

        # selection phase
        for i in range(theta):
            aggregated_grad = self.sel_agg.agg_g(grads[mask])
            closest_idx = self.find_closest_grad(grads, aggregated_grad, mask)
            # index of client
            logger.log_value('bulyan/select', closest_idx, optimizer.niters)
            selection_set.append(grads[closest_idx])

            # removing the closest vector
            mask[closest_idx] = False

        return self.agg_agg.agg_g(torch.stack(selection_set))

    def __str__(self):
        return f"<BulyanP1 with selector aggregator {self.sel_agg} and final aggregator {self.agg_agg}>"


class BulyanWithMultiKurm(Aggregator):
    # Same as bulyan but without phase 2

    def __init__(self, logger, optimizer, world_size, num_byz=0, attack=None,
                 grad_hook=None, agg_agg=None):
        super().__init__(logger, optimizer, world_size, num_byz=num_byz,
                         attack=attack, grad_hook=grad_hook)

        n = self.world_size
        f = self.num_byz
        theta = n - 2 * f

        self.sel_agg = Krum(logger, optimizer, num_byz, attack,
                            grad_hook, num_krum=theta)
        self.agg_agg = agg_agg

    def agg_g(self, grads):
        # aggregation rule implementation according to section 4 of the
        # paper

        n = self.world_size
        f = self.num_byz
        theta = n - 2 * f

        assert n - 4 * f - 3 >= 0

        selection_set = []
        received_set = grads

        # selection phase
        selection_set = self.sel_agg.agg_g(received_set)

        # aggregation phase
        return self.agg_agg.agg_g(selection_set)

    def __str__(self):
        return f"<BulyanWithMultiKrum with selector {self.sel_agg} and with aggregator {self.agg_agg}>"


class Stochastic(Aggregator):
    def __init__(self, logger, optimizer, world_size, num_byz=0, attack=None,
                 grad_hook=None, probs=None, aggregators=None):
        super().__init__(logger, optimizer, world_size, num_byz=num_byz,
                         attack=attack, grad_hook=grad_hook)

        self.aggregators = aggregators
        self.probs = [probs[0]]
        for prob in probs[1:]:
            self.probs.append(self.probs[-1] + prob)

    def agg_g(self, grads):
        random = torch.rand((1,))
        #position = bisect.bisect_left(self.probs, random)
        optimizer = self.optimizer
        np.random.shuffle(self.aggregators)
        # self.logger.log_value('stochastic', position, optimizer.niters)
        #self.logger.info(f'Chose {self.aggregators[0]}!')
        return self.aggregators[0].agg_g(grads)


class OmniScient(Aggregator):
    def agg_g(self, grads):
        num_nonbyz = self.world_size - self.num_byz
        return torch.mean(grads[:num_nonbyz], dim=0)
