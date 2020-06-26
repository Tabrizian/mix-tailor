from __future__ import print_function
import os
import argparse
import dbqf.wg.cluster
import dbqf.exps as exps
import dbqf.exps.cifar10
import dbqf.exps.mnist
import dbqf.exps.rebuttal


class RunSingle(object):
    def __init__(self, log_dir, exclude, prefix, parallel=False):
        self.log_dir = log_dir
        self.num = 0
        self.exclude = exclude
        self.parallel = parallel
        self.prefix = prefix

    def __call__(self, args):
        logger_name = 'runs/%s/%s_%03d_' % (self.log_dir,
                                            self.prefix, self.num)
        cmd = ['python3 -m dbqf.main']
        self.num += 1
        for k, v in args:
            if v is not None:
                cmd += ['--{} {}'.format(k, v)]
                if k not in self.exclude:
                    logger_name += '{}_{},'.format(k, v)
        dir_name = logger_name.strip(',')
        cmd += ['--log-dir "$dir_name"']
        cmd += ['> "$dir_name/node-$SLURM_PROCID.log" 2>&1']
        cmd = ['export dir_name="%s"; mkdir -p "$dir_name" && ' % dir_name] + cmd
        if self.parallel:
            cmd += ['&']
        return ' '.join(cmd)


def deep_product(args, index=0, cur_args=[]):
    if index >= len(args):
        yield cur_args
    elif isinstance(args, list):
        # Disjoint
        for a in args:
            for b in deep_product(a):
                yield b
    elif isinstance(args, tuple):
        # Disjoint product
        for a in deep_product(args[index]):
            next_args = cur_args + a
            for b in deep_product(args, index+1, next_args):
                yield b
    elif isinstance(args, dict):
        # Product
        keys = list(args.keys())
        values = list(args.values())
        if not isinstance(values[index], list):
            values[index] = [values[index]]
        for v in values[index]:
            if not isinstance(v, tuple):
                next_args = cur_args + [(keys[index], v)]
                for a in deep_product(args, index+1, next_args):
                    yield a
            else:
                for dv in deep_product(v[1]):
                    next_args = cur_args + [(keys[index], v[0])]
                    next_args += dv
                    for a in deep_product(args, index+1, next_args):
                        yield a


def run_multi(run_single, args):
    cmds = []
    for arg in deep_product(args):
        cmds += [run_single(arg)]
    return cmds


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--grid', default='gvar', type=str)
    parser.add_argument('--run_name', default='', type=str)
    parser.add_argument('--cluster', default='bolt', type=str)
    parser.add_argument('--cluster_args', default='8,4,gpu', type=str)
    parser.add_argument('--run0_id', default=0, type=int)
    parser.add_argument('--prefix', default='0', type=str)
    args = parser.parse_args()
    prefix = args.prefix
    run0_id = args.run0_id
    val = exps.__dict__[args.grid].__dict__[args.run_name]([])
    args_run, log_dir, exclude = val
    parallel = True

    run_single = RunSingle(log_dir, exclude, prefix, True)
    run_single.num = run0_id

    cmds = run_multi(run_single, args_run)
    jobs = dbqf.wg.cluster.__dict__[
        args.cluster](args.cluster_args, args.prefix, len(cmds))

    for j, job_index in enumerate(jobs):
        file_name = f'jobs/{prefix}/{job_index}.sh'.format()
        try:
            os.mkdir(f'jobs/{prefix}')
        except OSError:
            pass
        with open(file_name, 'w') as f:
            for i in range(j, len(cmds), len(jobs)):
                print(cmds[i], file=f)
            if parallel:
                print('wait', file=f)
