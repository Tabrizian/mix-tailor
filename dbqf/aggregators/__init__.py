from dbqf.aggregators.aggregators import *
import numpy as np


def get_aggregator(opt, optim, logger):
    aggregator = None
    if opt.aggregator == 'bulyan':
        bul_sel = get_aggregator_by_name(opt,
                                         opt.bul_sel,
                                         optim,
                                         logger
                                         )
        bul_agg = get_aggregator_by_name(opt,
                                         opt.bul_agg,
                                         optim,
                                         logger
                                         )
        aggregator = Bulyan(logger, optim, opt.world_size, opt.num_byz, None, None,
                            bul_sel, bul_agg)
    elif opt.aggregator == 'bulyanp1':
        bul_sel = get_aggregator_by_name(opt,
                                         opt.bul_sel,
                                         optim,
                                         logger
                                         )
        bul_agg = get_aggregator_by_name(opt,
                                         opt.bul_agg,
                                         optim,
                                         logger
                                         )
        aggregator = Bulyan(logger, optim, opt.world_size, opt.num_byz, None, None,
                            bul_sel, bul_agg)
    elif opt.aggregator == 'bulyanmkrum':
        bul_agg = get_aggregator_by_name(opt,
                                         opt.bul_agg,
                                         optim,
                                         logger
                                         )
        aggregator = BulyanWithMultiKurm(logger, optim, opt.world_size, opt.num_byz, None,
                                         None, bul_agg)

    elif opt.aggregator == 'stochastic':
        aggregators = []
        num_stochastic = 16
        requested_aggregators = opt.stochastic_aggs.split(',')
        norms = np.arange(1, num_stochastic + 1)
        if 'krum' in requested_aggregators:
            for norm in norms:
                local_aggregator = Krum(logger, optim, opt.world_size, opt.num_byz, None, norm=norm)
                aggregators.append(local_aggregator)
        if 'comed' in requested_aggregators:
            for norm in norms:
                local_aggregator = CoMed(logger, optim, opt.world_size, opt.num_byz, None)
                aggregators.append(local_aggregator)
        if 'geomed' in requested_aggregators:
            for norm in norms:
                local_aggregator = GeoMed(logger, optim, opt.world_size, opt.num_byz, None, norm=norm)
                aggregators.append(local_aggregator)

        if 'bulyan' in requested_aggregators:
            bul_sels = ['krum', 'geomed', 'comed', 'fedavg']
            bul_aggs = ['krum', 'geomed', 'comed', 'fedavg']
            for bul_sel in bul_sels:
                for bul_agg in bul_aggs:
                    opt.aggregator = 'bulyan'
                    opt.bul_sel = bul_sel
                    opt.bul_agg = bul_agg
                    local_aggregator = get_aggregator(opt, optim, logger)
                    aggregators.append(local_aggregator)

        np.random.shuffle(aggregators)
        probabilites = []
        for i in range(len(aggregators)):
            probabilites.append(1.0 / len(aggregators))

        aggregator = Stochastic(logger, optim, opt.world_size, opt.num_byz, None,
                                aggregators=aggregators, probs=probabilites)
        opt.aggregator = 'stochastic'
    elif opt.aggregator != 'bulyan':
        aggregator = get_aggregator_by_name(opt,
                                            opt.aggregator,
                                            optim,
                                            logger
                                            )

    return aggregator


def get_aggregator_by_name(opt, name, optim, logger):

    aggregator = None

    if name == 'fedavg':
        aggregator = FedAvg(logger, optim, opt.world_size, opt.num_byz)
    elif name == 'krum':
        aggregator = Krum(logger, optim, opt.world_size, opt.num_byz)
    elif name == 'geomed':
        aggregator = GeoMed(logger, optim, opt.world_size, opt.num_byz)
    elif name == 'comed':
        aggregator = CoMed(logger, optim, opt.world_size, opt.num_byz)
    elif name == 'omniscient':
        aggregator = OmniScient(logger, optim, opt.world_size, opt.num_byz)
    else:
        raise Exception(f'Not found name {name}')

    return aggregator
