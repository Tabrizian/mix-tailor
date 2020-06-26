import argparse


def parse_args():
    parser = argparse.ArgumentParser('distributed training')
    parser.add_argument('--dist-url', help='master URL')
    parser.add_argument('--world-size', help='size of the world', type=int)
    parser.add_argument('--rank', help='worker rank', type=int)
    parser.add_argument(
        '--agg-grad',
        help='aggregate gradients instead of weights',
        action='store_true')
    parser.add_argument('--seed', help='torch manual seed',
                        type=int, default=1)
    parser.add_argument('--lr', help='learning rate',
                        type=float, default=0.1)
    parser.add_argument('--data', help='data directory',
                        type=str, default='./data')
    parser.add_argument('--batch-size', help='batch size',
                        type=int, default=128)
    parser.add_argument('--test-batch-size', help='test batch size',
                        type=int, default=256)
    parser.add_argument('--epochs', help='number of epochs',
                        type=int, default=5)
    parser.add_argument('--gamma', help='gamma rate',
                        type=float, default=0.1)
    parser.add_argument('--log-interval', help='log interval',
                        type=int, default=10)
    parser.add_argument('--log-dir', help='log directory',
                        type=str)
    parser.add_argument('--num-class', help='number of classes',
                        type=int, default=10)
    parser.add_argument('--data-aug', help='data augmentation',
                        action='store_false')
    parser.add_argument('--cuda', help='run on CUDA',
                        action='store_true')
    parser.add_argument('--workers', help='number of data workers',
                        type=int)
    parser.add_argument('--model', help='number of data workers',
                        type=str)
    parser.add_argument('--arch', help='architecture',
                        type=str)
    parser.add_argument('--dataset', help='dataset',
                        type=str)
    parser.add_argument('--epoch-iters', help='epoch iterations',
                        type=int)
    parser.add_argument('--optim', help='optimizer',
                        type=str, default='sgd')
    parser.add_argument('--momentum', help='momentum',
                        type=float)
    parser.add_argument('--nodropout', help='no dropout',
                        action='store_true')
    parser.add_argument('--no-transform', help='no transform',
                        action='store_true')
    parser.add_argument('--local-rounds', help='number of local rounds',
                        type=int)
    parser.add_argument('--weight-decay', help='weight decay',
                        type=float, default=0.0)
    parser.add_argument('--heter', help='use heterogenous',
                        action='store_true')
    parser.add_argument('--resampling', help='Number of times to resample', type=int, default=None)
    parser.add_argument('--global_rank', help='global rank of GPU',
                        type=int)
    parser.add_argument('--aggregator', help='distributed training aggregator')
    parser.add_argument('--stochastic-aggs', type=str, help='name of the stochastic aggregators')
    parser.add_argument('--adaptive-attack-eps', type=str, help='name of the epsilons used for the adaptive attack')
    parser.add_argument('--bul-sel',
                        help='bulyan aggregator for selection phase')
    parser.add_argument('--alittle-num',
                        type=float,
                        default=3,
                        help='bulyan aggregator for selection phase')
    parser.add_argument('--bul-agg',
                        help='bulyan aggregator for aggregation phase')
    parser.add_argument('--eps-reverse', type=float, default=0.1,
                        help='Epsilon for the reverse attack hyperparameter')
    parser.add_argument('--num-byz', help='number of byzantine workers',
                        type=int, default=0)
    parser.add_argument('--nesterov', help='nesterov', action='store_true')
    parser.add_argument('--niters', help='number of iterations',
                        type=int)
    parser.add_argument('--attack', help='type of the attack',
                        type=str)
    parser.add_argument('--distributed', help='running in distributed mode',
                        action='store_true')
    parser.add_argument('--pretrained', help='use pretrained model',
                        action='store_true')
    opt = parser.parse_args()
    return opt
