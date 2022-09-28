from scipy import interpolate
import collections
import numpy as np
import os
import re
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import math
import itertools
import tensorboard.backend.event_processing.event_accumulator as EA
from mpl_toolkits.mplot3d import Axes3D


def get_run_names(logdir, patterns):
    run_names = []
    for pattern in patterns:
        for root, subdirs, files in os.walk(logdir, followlinks=True):
            if re.match(pattern, root):
                run_names += [root]
    # print(run_names)
    run_names.sort()
    return run_names


def get_run_names_events_tb(run_names):
    # return the workers tensorboard file corresponding to a pattern
    # {'run1': {0: ['file1'], 1: ['file2'], 2: ['file3'], ...}}
    for key, value in run_names.items():
        path = key + '/' + value[0]
        num_workers = len(os.listdir(path))
        run_names[key] = {}
        for i in range(num_workers):
            run_names[key][i] = []
            for file in os.listdir(path + '/' + str(i)):
                run_names[key][i].append(file)
            run_names[key][i] = sorted(run_names[key][i])
    return run_names


def get_run_names_events(logdir, patterns):
    # get all the tb folders corresponding to the patterns
    run_names = {}
    for pattern in patterns:
        for root, subdirs, files in os.walk(logdir, followlinks=True):
            if re.match(pattern, root):
                for dir in subdirs:
                    if re.match('.*tb.*', dir):
                        if root not in run_names.keys():
                            run_names[root] = []
                        # TODO fix this for multi folder
                        if len(run_names[root]) == 0:
                            run_names[root].append(dir)
                if root in run_names.keys():
                    run_names[root] = sorted(run_names[root])

    return get_run_names_events_tb(run_names)


def calc_avg(data):
    grad = 0.0
    # find the common set of items in the workers array
    min_len = min([len(val[1]) for _, val in data.items()])
    i = 0
    for key, val in data.items():
        grad += val[1][:min_len]
        i += 1
    return (val[0][:min_len], grad / i)


def calc_total_avg(data):
    # calculate average for every key
    avg_data = {}
    for key, val in data.items():
        if key not in avg_data:
            avg_data[key] = {}
        for second_key in val:
            avg_data[key][second_key] = calc_avg(val[second_key])
    return avg_data


def get_data_pth_events(logdir, run_names, tag_names, batch_size=None):
    data = []
    all_points = []
    for run_name, workers in run_names.items():
        d = {}
        points = {}

        for worker, events in workers.items():

            for event in events:
                path = run_name+'/tb/'+str(worker) + '/' + event
                ea = EA.EventAccumulator(path,
                                         size_guidance={
                                             EA.COMPRESSED_HISTOGRAMS: 500,
                                             EA.IMAGES: 4,
                                             EA.AUDIO: 4,
                                             EA.SCALARS: 0,
                                             EA.HISTOGRAMS: 1,
                                         })
                ea.Reload()
                for tag_name in tag_names:
                    if tag_name not in ea.Tags()['scalars']:
                        continue
                    scalar = ea.Scalars(tag_name)
                    if tag_name not in d:
                        d[tag_name] = np.array(
                            [[dp.step for dp in scalar], [dp.value for dp in scalar]])
                        points[tag_name] = [len(d[tag_name][0]) - 1]
                    else:
                        new_array = np.array([dp.step for dp in scalar])
                        indexes = new_array > d[tag_name][0][-1]
                        res1 = np.concatenate(
                            (d[tag_name][0], np.array([dp.step for dp in scalar])[indexes]))
                        res2 = np.concatenate(
                            (d[tag_name][1], np.array([dp.value for dp in scalar])[indexes]))
                        points[tag_name].append(len(res2) - 1)
                        d[tag_name] = (res1, res2)
        data += [d]
        all_points += [points]

    return data, all_points


def plot_smooth(x, y, npts=100, order=3, points=None, vlines=None, *args, **kwargs):
    # points = np.array(points, dtype=int)
    # plt.plot(x[points], y[points], 'o',  )

    x_smooth = np.linspace(x.min(), x.max(), npts)
    tck = interpolate.splrep(x, y, k=order)
    y_smooth = interpolate.splev(x_smooth, tck, der=0)
# 
    plt.plot(x_smooth, y_smooth, *args, **kwargs)
    plt.ticklabel_format(axis="x", style="sci", scilimits=None)


def plot_bar(x, points=None, vlines=None, *args, **kwargs):
    print(args)
    print(kwargs)
    tag_name = list(x)[0]
    plt.hist(x[tag_name][1].astype(np.int32), np.arange(0, 13, 1), histtype='step', stacked=True, fill=False, *args, **kwargs)


def plot_smooth_o1(x, points=None, vlines=None, *args, **kwargs):
    plot_smooth(x[0], x[1], 100, 1, points, vlines, *args, **kwargs)


def get_legend(lg_tags, run_name, lg_replace=[]):
    lg = ""
    for lgt in lg_tags:
        res = ".*?($|,)" if ',' not in lgt and '$' not in lgt else ''
        mg = re.search(lgt + res, run_name)
        if mg:
            lg += mg.group(0)
    lg = lg.replace('_,', ',')
    lg = lg.strip(',')
    for a, b in lg_replace:
        lg = lg.replace(a, b)
    return lg


class OOMFormatter(mtick.ScalarFormatter):
    def __init__(self, useOffset=None, useMathText=None, useLocale=None, acc_bits=None):
        super().__init__(useOffset=useOffset, useMathText=useMathText, useLocale=useLocale)
        if acc_bits is not None:
            self.acc_bits = acc_bits
        else:
            self.acc_bits = 3

    def __call__(self, x, pos=None):
        """
        Return the format for tick value *x* at position *pos*.
        """
        if len(self.locs) == 0:
            return ''
        else:
            xp = (x - self.offset) / (10. ** self.orderOfMagnitude)
            if abs(xp) < 1e-8:
                xp = 0
            if self._useLocale:
                s = locale.format_string(self.format, (xp,))
            else:
                s = self.format % xp
            return self.fix_minus(s)

    def _set_format(self):
        bits = self.acc_bits
        # set the format string to format all the ticklabels
        if len(self.locs) < 2:
            # Temporarily augment the locations with the axis end points.
            _locs = [*self.locs, *self.axis.get_view_interval()]
        else:
            _locs = self.locs
        locs = (np.asarray(_locs) - self.offset) / 10. ** self.orderOfMagnitude
        loc_range = np.ptp(locs)
        # Curvilinear coordinates can yield two identical points.
        if loc_range == 0:
            loc_range = np.max(np.abs(locs))
        # Both points might be zero.
        if loc_range == 0:
            loc_range = 1
        if len(self.locs) < 2:
            # We needed the end points only for the loc_range calculation.
            locs = locs[:-2]
        loc_range_oom = int(math.floor(math.log10(loc_range)))
        # first estimate:
        sigfigs = max(0, bits - loc_range_oom)
        # refined estimate:
        thresh = 10 ** (-bits) * 10 ** (loc_range_oom)
        while sigfigs >= 0:
            if np.abs(locs - np.round(locs, decimals=sigfigs)).max() < thresh:
                sigfigs -= 1
            else:
                break
        sigfigs = bits
        self.format = '%1.' + str(sigfigs) + 'f'
        if self._usetex or self._useMathText:
            self.format = r'$\mathdefault{%s}$' % self.format


def plot_3d(x, points=None, vlines=None, *args, **kwargs):
    print(args)
    print(kwargs)
    workers = {}
    for key, value in x.items():
        key_splited = key.split('/')

        worker_id = key_splited[1]
        if worker_id not in workers:
            workers[worker_id] = {}
        dim_id = key_splited[2]
        workers[worker_id][int(dim_id)] = value[1]

    for worker, value in workers.items():
        result = np.zeros((len(value.values()), len(value[0])))
        for key, val in value.items():
            result[key] = val
        workers[worker] = result

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for worker in workers:
        point = workers[worker][:, 20]
        if int(worker) < 10:
            ax.scatter(point[0], point[1], point[2], color='b')
        else:
            ax.scatter(point[0], point[1], point[2], color='r')


def plot_tag(data, plot_f, run_names, tag_name, lg_tags, ylim=None, color0=0,
             ncolor=None, lg_replace=[], no_title=False, points=None, xlim=None, vlines=None, orders=None, acc_bits=None, markeroff=True):
    xlabel = {'krum/select': 'Client Index', 'bulyan/select': 'Client Index', 'stochastic': 'Aggregator Index'}
    ylabel = {'Tacc': 'Training Accuracy (%)', 'Terror': 'Training Error (%)',
              'train/accuracy': 'Training Accuracy (%)',
              'Vacc': 'Test Accuracy (%)', 'Verror': 'Test Error (%)',
              'valid/accuracy': 'Test Accuracy (%)',
              'loss': 'Loss',
              'epoch': 'Epoch',
              'Tloss': 'Loss', 'Vloss': 'Loss', 'lr': 'Learning rate',
              'grad_bias': 'Gradient Diff norm',
              'est_var': 'Average Variance',
              'est_snr': 'Mean SNR',
              'nb_error': 'NB Error',
              'est_nvar': 'Mean Normalized Variance',
              'krum/select': 'Client Selection Frequency',
              'tsne/': 'TSNE',
              'stochastic': 'Aggregator Selection Frequency',
              'bulyan/select': 'Client Selection Frequency'}
    titles = {'Tacc': 'Training Accuracy', 'Terror': 'Training Error',
              'train/accuracy': 'Training Accuracy',
              'Vacc': 'Test Accuracy', 'Verror': 'Test Error',
              'loss': 'Loss',
              'epoch': 'Epoch',
              'Tloss': 'Loss on full training set', 'lr': 'Learning rate',
              'Vloss': 'Loss on validation set',
              'grad_bias': 'Optimization Step Bias',
              'nb_error': 'Norm-based Variance Error',
              'est_var': 'Optimization Step Variance',
              'est_snr': 'Optimization Step SNR',
              'est_nvar': 'Optimization Step Normalized Variance (w/o lr)',
              'krum/select': 'Krum Client Selection',
              'bulyan/select': 'Bulyan Client Selection',
              'stochastic': 'Stochastic Aggregator Selection'
              }

    yscale_log = ['Tloss', 'est_var', 'krum/select', 'bulyan/select']  # , 'est_var'
    yscale_log_offset = ['Vloss']  # , 'est_var'
    yscale_scalar = ['Vloss']  # , 'est_var'

    yscale_base = []
    # yscale_sci = ['est_bias', 'est_var']
    plot_fs = {'Tacc': plot_f, 'Vacc': plot_f,
               'Terror': plot_f, 'Verror': plot_f,
               'Tloss': plot_f, 'Vloss': plot_f,
               'loss': plot_f, 'krum/select': plot_bar,
               'bulyan/select': plot_bar,
               'stochastic': plot_bar,
               'tsne/.*': plot_3d
               }

    plotting_function = None
    for j in list(plot_fs.keys()):
        if re.match(j, tag_name):
            plotting_function = plot_fs[j]
    if tag_name not in xlabel:
        xlabel[tag_name] = 'Training Iteration'
    if plotting_function is None:
        plot_fs[tag_name] = plot_f
    else:
        plot_fs[tag_name] = plotting_function

    if not isinstance(data, list):
        data = [data]
        run_names = [run_names]

    color = ['blue', 'orangered', 'darkred', 'darkkhaki', 'darkblue', 'grey']
    # color = color[:ncolor]
    style = ['-', '--', ':', '-.']
    #style = ['-']
    # color = [[0.00784314, 0.24313725, 1.],
    #          [1., 0.48627451, 0.],
    #          [0.10196078, 0.78823529, 0.21960784],
    #          [0.90980392, 0., 0.04313725],
    #          [0.54509804, 0.16862745, 0.88627451]]
    #style = ['-', '--', ':', '-.']
    styles = ['-']

#     markers =
    colors = color
#     styles = ['-', '--', ':', '-.']
    markers = ['o', 'X', 'p', '*', 'd', 'v']
    plt.rcParams.update({'font.size': 16})
    plt.grid(linewidth=1)
    legends = []
    # extract run index
    indexes = [int(run_names[i].split('/')[-1].split('_')[1])
               for i in range(len(run_names))]
    s_indexes = np.argsort(indexes)

    for i in range(len(data)):
        if tag_name not in data[i]:
            continue
        legends += [get_legend(lg_tags, run_names[i], lg_replace)]
        if orders:
            color_index = orders.index(legends[-1])
        else:
            color_index = color0 + i
        if not markeroff:
            plot_fs[tag_name](
                data[i][tag_name], None,
                vlines=vlines,
                linestyle=style[0], label=legends[-1],
                color=color[(color_index) % len(color)], linewidth=2, marker=markers[(color_index) % len(markers)], markersize=10, markevery=10 + 2*(color_index % 5))
        else:
            plot_fs[tag_name](
                data[i][tag_name], None,
                vlines=vlines,
                linestyle=style[int((color_index) / len(color))], label=legends[-1],
                color=color[(color_index) % len(color)], linewidth=2)
    if not no_title:
        plt.title(titles[tag_name])
    if tag_name in yscale_log:
        ax = plt.gca()
        if tag_name in yscale_base:
            ax.set_yscale('log', basey=np.e)
            ax.yaxis.set_major_formatter(mtick.FuncFormatter(ticks))
        elif tag_name in yscale_log_offset:
            ax.set_yscale('log')
            ax.yaxis.set_major_formatter(
                mtick.ScalarFormatter(useOffset=True))
            ax.yaxis.set_major_formatter(
                mtick.ScalarFormatter(useOffset=True))
            ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
            print('Hone')
        else:
            ax.set_yscale('log')
    else:
        ax = plt.gca()
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3))

    if tag_name in yscale_scalar:
        ax.set_yscale('log')
        ax.yaxis.set_minor_locator(mtick.LogLocator(base=10.0, subs=[2, 4, 6]))
        ax.yaxis.set_minor_formatter(mtick.ScalarFormatter())
        ax.yaxis.set_major_formatter(OOMFormatter(acc_bits=1))
        #ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

    ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    if ylim is not None:
        plt.ylim(ylim)
    handles, labels = plt.gca().get_legend_handles_labels()

    if orders:
        norders = []
        for order in orders:
            if order in labels:
                norders.append(order)

        order = []

        for label in labels:
            order.append(norders.index(label))

        nlabels = np.arange(len(labels)).tolist()
        nhandles = np.arange(len(handles)).tolist()
        for idx, label, handle in zip(order, labels, handles):
            nlabels[idx] = label
            nhandles[idx] = handle
        handles = nhandles
        labels = nlabels

    plt.legend(handles, labels,
               loc="upper left", bbox_to_anchor=(1.01, 1.0), prop={'size': 12})
    if vlines:
        for vline in vlines:
            plt.axvline(vline, linestyle='--', color='black')
    if xlim:
        plt.xlim(xlim)
    plt.xlabel(xlabel[tag_name])
    plt.ylabel(ylabel[tag_name])


def ticks(y, pos):
    return r'$e^{{{:.0f}}}$'.format(np.log(y))


def ticks_10(y, pos):
    return r'${0:g}$'.format(np.log10(y))


def plot_runs_and_tags(get_data_f, plot_f, logdir, patterns, tag_names,
                       fig_name, lg_tags, ylim, batch_size=None, sep_h=True,
                       ncolor=None, save_single=False, lg_replace=[],
                       xlim=None, acc_bits=None, markeroff=True,
                       no_title=False, vlines=None, color_order=None):
    run_names = get_run_names_events(logdir, patterns)
    data, points = get_data_f(logdir, run_names, tag_names, batch_size)
    if len(data) == 0:
        return data, run_names
    tag_names = list(data[0])
    num = len(tag_names)
    height = (num + 1) // 2
    width = 2 if num > 1 else 1
    if not save_single:
        fig = plt.figure(figsize=(7 * width, 4 * height))
        fig.subplots(height, width)
    else:
        plt.figure(figsize=(7, 4))
    plt.tight_layout(pad=1., w_pad=3., h_pad=3.0)
    fi = 1
    if save_single:
        fig_dir = fig_name[:fig_name.rfind('.')]
        try:
            os.makedirs(fig_dir)
        except os.error:
            pass
    for i in range(len(tag_names)):
        yl = ylim[i]
        if not isinstance(yl, list) and yl is not None:
            yl = ylim
        if not save_single:
            plt.subplot(height, width, fi)
        plot_tag(data, plot_f, list(run_names), tag_names[i], lg_tags, yl,
                 ncolor=ncolor, lg_replace=lg_replace, no_title=no_title, points=points, vlines=vlines, xlim=xlim, orders=color_order,
                 acc_bits=acc_bits, markeroff=markeroff)
        if save_single:
            if len(tag_names[i].split('/')) > 1:
                try:
                    os.makedirs(fig_dir + '/' + tag_names[i])
                except os.error:
                    pass

            plt.savefig('%s/%s-lo.pdf' % (fig_dir, tag_names[i]),
                        dpi=100, bbox_inches='tight')

            ax = plt.gca()

            handles, labels = ax.get_legend_handles_labels()
            if color_order:
                norders = []
                for order in color_order:
                    if order in labels:
                        norders.append(order)

                order = []

                for label in labels:
                    order.append(norders.index(label))

                nlabels = np.arange(len(labels)).tolist()
                nhandles = np.arange(len(handles)).tolist()
                for idx, label, handle in zip(order, labels, handles):
                    nlabels[idx] = label
                    nhandles[idx] = handle
                handles = nhandles
                labels = nlabels

            plt.legend(handles, labels, prop={'size': 12})

            plt.savefig('%s/%s-li.pdf' % (fig_dir, tag_names[i]),
                        dpi=100, bbox_inches='tight')

            ax.get_legend().remove()
            plt.savefig('%s/%s.pdf' % (fig_dir, tag_names[i]),
                        dpi=100, bbox_inches='tight')

            plt.figure(figsize=(7, 4))
        fi += 1
    plt.savefig(fig_name, dpi=100, bbox_inches='tight')
    return data, run_names


def find_largest_common_iteration(iters):
    intersect = set(iters[0])
    for i in range(1, len(iters)):
        intersect = intersect & set(iters[i])
    return list(intersect)
