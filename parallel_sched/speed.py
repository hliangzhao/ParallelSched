"""
Fit a speed function for each model.
"""
import numpy as np
import ast
import scipy.interpolate


def fit():
    funcs = dict()
    records = []

    with open("config_speed.txt", "r") as f:
        for line in f:
            records.append(ast.literal_eval(line.replace("\n", "")))
        speed_maps = dict()

        for record in records:
            model, sync_mode, total_batch_size, num_ps, num_worker, speeds, ps_cpu_usages, worker_cpu_usages = record
            if model not in speed_maps:
                speed_maps[model] = []
            speed_maps[model].append((num_ps, num_worker, sum(speeds)))

        for model in speed_maps.keys():
            x, y, z = [], [], []
            for _num_ps, _num_worker, _speed in speed_maps[model]:
                x.append(_num_ps)
                y.append(_num_worker)
                z.append(_speed)
            func = scipy.interpolate.Rbf(np.array(x), np.array(y), np.array(z), function="linear")
            funcs[model] = func

        return funcs


speed_funcs = fit()
