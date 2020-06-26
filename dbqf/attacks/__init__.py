from dbqf.attacks.attacks import *
from dbqf.aggregators import get_aggregator
import numpy as np


def get_attack(opt, grad_size):
    if opt.attack is not None:
        if opt.attack == 'gaussian':
            return Gaussian(opt.num_byz, grad_size)
        elif opt.attack == 'mimic':
            return Mimic(opt.num_byz, grad_size)
        elif opt.attack == 'reverse':
            return Reverse(opt.num_byz, grad_size)
        elif opt.attack == 'alittle':
            return ALittle(opt.num_byz, grad_size, opt.alittle_num)
        elif opt.attack == 'epsilonreverse':
            return EpsilonReverse(opt.num_byz, grad_size, opt.eps_reverse)
        elif opt.attack == 'adaptive':
            aggregators = []
            if opt.aggregator == 'statistic':
                num_stochastic = 16
                requested_aggregators = opt.stochastic_aggs.split(',')
                norms = np.arange(1, num_stochastic + 1)
                if 'krum' in requested_aggregators:
                    for norm in norms:
                        local_aggregator = Krum(
                            None, None, opt.world_size, opt.num_byz, None, norm=norm)
                        aggregators.append(local_aggregator)
                if 'comed' in requested_aggregators:
                    for norm in norms:
                        local_aggregator = CoMed(
                            None, None, opt.world_size, opt.num_byz, None)
                        aggregators.append(local_aggregator)
                if 'geomed' in requested_aggregators:
                    for norm in norms:
                        local_aggregator = GeoMed(
                            None, None, opt.world_size, opt.num_byz, None, norm=norm)
                        aggregators.append(local_aggregator)

                if 'bulyan' in requested_aggregators:
                    bul_sels = ['krum', 'geomed', 'comed', 'fedavg']
                    bul_aggs = ['krum', 'geomed', 'comed', 'fedavg']
                    for bul_sel in bul_sels:
                        for bul_agg in bul_aggs:
                            opt.aggregator = 'bulyan'
                            opt.bul_sel = bul_sel
                            opt.bul_agg = bul_agg
                            local_aggregator = get_aggregator(opt, None, None)
                            aggregators.append(local_aggregator)
            else:
                aggregators.append(get_aggregator(opt, None, None))

            requested_eps = opt.adaptive_attack_eps.split(',')
            attacks = []
            for requested_epsilon in requested_eps:
                attacks.append(EpsilonReverse(
                    opt.num_byz, grad_size, float(requested_epsilon)))

            return Adaptive(opt.num_byz, grad_size, attacks, aggregators)

    return None
