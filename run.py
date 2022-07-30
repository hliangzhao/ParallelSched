"""
The entry.
"""
import os
import sys
import datetime
import copy
import time
import numpy as np
import multiprocessing
import parameter_template as pt

sl_cfg = {
    "TRAINING_MODE": "SL",
    "VALUE_NET": False,
    "POLICY_NN_MODEL": None,
    "VALUE_NN_MODEL": None,
    "CHECKPOINT_INTERVAL": 50,
    "LEARNING_RATE": 0.005,
    "TOTAL_NUM_STEPS": 200,
    "VAL_INTERVAL": 50,
    "NUM_TS_PER_UPDATE": 5,
    "JOB_ORDER_SHUFFLE": True
}

NUM_TEST = 5
PARALLELISM = 10
TASK_ID = -1


def replace_params(m, dir_path):
    pt_module = globals().get("pt", None)
    train_cfg = dict()
    if pt_module:
        train_cfg = {
            k: v for k, v in pt_module.__dict__.items() if not (k.startswith("__") or k.startswith("_"))
        }

    f = open(dir_path, "parameter.py", "w")
    for k, _ in train_cfg.items():
        if k in m.keys():
            train_cfg[k] = m[k]
        if isinstance(train_cfg[k], str):
            f.write(str(k) + " = '" + str(train_cfg[k]) + "'\n")
        else:
            f.write(str(k) + " = " + str(train_cfg[k] + "\n"))
    f.close()


def get_cfg(id, exp_name, test_val):
    cfg = dict()
    cfg["EXPERIMENT_NAME"] = exp_name + "_" + str(test_val)
    if id == 1:
        cfg["SCHED_WINDOW_SIZE"] = test_val
        cfg["STATE_DIM"] = (sum([enable for (_, enable) in pt.INPUTS_GATE]), test_val)
        cfg["ACTION_DIM"] = 3 * test_val + pt.SKIP_TS
        cfg["NUM_NEURONS_PER_FCN"] = sum([enable for (_, enable) in pt.INPUTS_GATE]) * test_val
    elif id == 2:
        cfg["NUM_FCN_LAYERS"] = 1
        cfg["NUM_NEURONS_PER_FCN"] = test_val
    elif id == 3 or id == 24:
        cfg["NUM_FCN_LAYERS"] = test_val
        cfg["NUM_NEURONS_PER_FCN"] = pt.STATE_DIM[0] * pt.STATE_DIM[1] * 2 / 3
    elif id == 4:
        cfg["BUNDLE_ACTION"] = test_val
        if test_val is False:
            cfg["ACTION_DIM"] = 2 * pt.SCHED_WINDOW_SIZE + pt.SKIP_TS
    elif id == 5:
        cfg["JOB_ARRIVAL_PATTERN"] = test_val
    elif id == 6:
        cfg["BATCH_NORMALIZATION"] = test_val
    elif id == 7:
        cfg["SL_LOSS_FUNCTION"] = test_val
    elif id == 8:
        # ["Norm_Progress", "Job_Progress", "Num_Uncompleted_Jobs"]
        if test_val == "Norm_Progress":
            cfg["TS_REWARD_PLUS_JOB_REWARD"] = False
            cfg["NUM_UNCOMPLETED_JOB_REWARD"] = False
        elif test_val == "Job_Progress":
            cfg["TS_REWARD_PLUS_JOB_REWARD"] = True
            cfg["NUM_UNCOMPLETED_JOB_REWARD"] = False
        elif test_val == "Num_Uncompleted_Jobs":
            cfg["TS_REWARD_PLUS_JOB_REWARD"] = False
            cfg["NUM_UNCOMPLETED_JOB_REWARD"] = True
    elif id == 9:
        if not test_val:
            cfg["REPLAY_MEMORY_SIZE"] = 256
    elif id == 10:
        cfg["VALUE_NET"] = test_val
    elif id == 11:
        if test_val:
            cfg["INJECT_SAMPLES"] = True
            cfg["EPSILON_GREEDY"] = False
        else:
            cfg["INJECT_SAMPLES"] = False
            cfg["EPSILON_GREEDY"] = True
    elif id == 12:
        cfg["JOB_ARRIVAL_PATTERN"] = test_val
        cfg["HEURISTIC"] = "DRF"
    elif id == 13:
        cfg["JOB_ARRIVAL_PATTERN"] = test_val
        cfg["HEURISTIC"] = "SRTF"
    elif id == 14:
        cfg["JOB_ARRIVAL_PATTERN"] = test_val
        cfg["HEURISTIC"] = "Tetris"
    elif id == 15:
        cfg["JOB_ARRIVAL_PATTERN"] = test_val
        cfg["HEURISTIC"] = "Optimus"
    elif id == 16:
        cfg["HEURISTIC"] = test_val
        cfg["MAX_NUM_WORKERS"] = 8
    elif id == 17:
        cfg["NUM_AGENTS"] = test_val
        cfg["MINI_BATCH_SIZE"] = 256 / test_val
    elif id == 18:
        cfg["CHANGING_JOB_TYPES"] = test_val
    elif id == 19:
        cfg["REAL_SPEED_TRACE"] = test_val
    elif id == 20:
        if test_val == "testbed":
            cfg["TESTBED"] = True
            cfg["CLUSTER_NUM_NODES"] = 6
            cfg["TOT_NUM_JOBS"] = 10
            cfg["MAX_NUM_EPOCHS"] = 1000
            cfg["MAX_ARRVS_PER_TS"] = 5
            cfg["TS_DURATION"] = 300.0
            window_size = 4
            cfg["SCHED_WINDOW_SIZE"] = window_size
            cfg["STATE_DIM"] = (sum([enable for (_, enable) in pt.INPUTS_GATE]), window_size)
            cfg["ACTION_DIM"] = 3 * window_size + pt.SKIP_TS
            cfg["NUM_NEURONS_PER_FCN"] = sum([enable for (_, enable) in pt.INPUTS_GATE]) * window_size
        elif test_val == "large-1":
            cfg["LARGE_SCALE"] = True
            cfg["CLUSTER_NUM_NODES"] = 100
            cfg["TOT_NUM_JOBS"] = 120
            cfg["MAX_NUM_EPOCHS"] = 80000
            cfg["MAX_ARRVS_PER_TS"] = 6
            cfg["TS_DURATION"] = 1200.0
            window_size = 30
            cfg["SCHED_WINDOW_SIZE"] = window_size
            cfg["STATE_DIM"] = (sum([enable for (_, enable) in pt.INPUTS_GATE]), window_size)
            cfg["ACTION_DIM"] = 3 * window_size + pt.SKIP_TS
            cfg["NUM_NEURONS_PER_FCN"] = sum([enable for (_, enable) in pt.INPUTS_GATE]) * window_size
        elif test_val == "large-2":
            cfg["LARGE_SCALE"] = True
            cfg["CLUSTER_NUM_NODES"] = 100
            cfg["TOT_NUM_JOBS"] = 180
            cfg["MAX_NUM_EPOCHS"] = 80000
            cfg["MAX_ARRVS_PER_TS"] = 9
            cfg["TS_DURATION"] = 1200.0
            window_size = 36
            cfg["SCHED_WINDOW_SIZE"] = window_size
            cfg["STATE_DIM"] = (sum([enable for (_, enable) in pt.INPUTS_GATE]), window_size)
            cfg["ACTION_DIM"] = 3 * window_size + pt.SKIP_TS
            cfg["NUM_NEURONS_PER_FCN"] = sum([enable for (_, enable) in pt.INPUTS_GATE]) * window_size
        elif test_val == "large-3":
            cfg["LARGE_SCALE"] = True
            cfg["CLUSTER_NUM_NODES"] = 120
            cfg["TOT_NUM_JOBS"] = 180
            cfg["MAX_NUM_EPOCHS"] = 80000
            cfg["MAX_ARRVS_PER_TS"] = 9
            cfg["TS_DURATION"] = 1200.0
            window_size = 36
            cfg["SCHED_WINDOW_SIZE"] = window_size
            cfg["STATE_DIM"] = (sum([enable for (_, enable) in pt.INPUTS_GATE]), window_size)
            cfg["ACTION_DIM"] = 3 * window_size + pt.SKIP_TS
            cfg["NUM_NEURONS_PER_FCN"] = sum([enable for (_, enable) in pt.INPUTS_GATE]) * window_size
        elif test_val == "large-4":
            cfg["LARGE_SCALE"] = True
            cfg["CLUSTER_NUM_NODES"] = 500
            cfg["TOT_NUM_JOBS"] = 600
            cfg["MAX_NUM_EPOCHS"] = 80000
            cfg["MAX_ARRVS_PER_TS"] = 30
            cfg["TS_DURATION"] = 1200.0
            cfg["MAX_NUM_WORKERS"] = 50
            window_size = 180
            cfg["SCHED_WINDOW_SIZE"] = window_size
            cfg["STATE_DIM"] = (sum([enable for (_, enable) in pt.INPUTS_GATE]), window_size)
            cfg["ACTION_DIM"] = 3 * window_size + pt.SKIP_TS
            cfg["NUM_NEURONS_PER_FCN"] = sum([enable for (_, enable) in pt.INPUTS_GATE]) * window_size
        elif test_val == "large-5":
            cfg["LARGE_SCALE"] = True
            cfg["CLUSTER_NUM_NODES"] = 500
            cfg["TOT_NUM_JOBS"] = 600
            cfg["MAX_NUM_EPOCHS"] = 80000
            cfg["MAX_ARRVS_PER_TS"] = 30
            cfg["TS_DURATION"] = 1200.0
            cfg["MAX_NUM_WORKERS"] = 100
            window_size = 180
            cfg["SCHED_WINDOW_SIZE"] = window_size
            cfg["STATE_DIM"] = (sum([enable for (_, enable) in pt.INPUTS_GATE]), window_size)
            cfg["ACTION_DIM"] = 3 * window_size + pt.SKIP_TS
            cfg["NUM_NEURONS_PER_FCN"] = sum([enable for (_, enable) in pt.INPUTS_GATE]) * window_size
        elif test_val == "large-6":
            cfg["LARGE_SCALE"] = True
            cfg["CLUSTER_NUM_NODES"] = 500
            cfg["TOT_NUM_JOBS"] = 600
            cfg["MAX_NUM_EPOCHS"] = 80000
            cfg["MAX_ARRVS_PER_TS"] = 30
            cfg["TS_DURATION"] = 1200.0
            cfg["MAX_NUM_WORKERS"] = 100
            cfg["VALUE_NET"] = False
            window_size = 180
            cfg["SCHED_WINDOW_SIZE"] = window_size
            cfg["STATE_DIM"] = (sum([enable for (_, enable) in pt.INPUTS_GATE]), window_size)
            cfg["ACTION_DIM"] = 3 * window_size + pt.SKIP_TS
            cfg["NUM_NEURONS_PER_FCN"] = sum([enable for (_, enable) in pt.INPUTS_GATE]) * window_size
        elif test_val == "small":  # by default
            cfg["CLUSTER_NUM_NODES"] = 48
            cfg["TOT_NUM_JOBS"] = 60
            cfg["MAX_NUM_EPOCHS"] = 80000
            cfg["MAX_ARRVS_PER_TS"] = 3
            cfg["TS_DURATION"] = 1200.0
            window_size = 20
            cfg["SCHED_WINDOW_SIZE"] = window_size
            cfg["STATE_DIM"] = (sum([enable for (_, enable) in pt.INPUTS_GATE]), window_size)
            cfg["ACTION_DIM"] = 3 * window_size + pt.SKIP_TS
            cfg["NUM_NEURONS_PER_FCN"] = sum([enable for (_, enable) in pt.INPUTS_GATE]) * window_size
    elif id == 21:
        cfg["JOB_RESR_BALANCE"] = test_val
    elif id == 22:
        if not test_val:
            cfg["POLICY_NN_MODEL"] = None
    elif id == 23:
        cfg["JOB_EPOCH_EST_ERROR"] = test_val
    elif id == 25:
        cfg["TRAIN_SPEED_ERROR"] = test_val
    return cfg


def process_results(root_dir, exp_name, test_vals):
    results = dict()
    for test_value in test_vals:
        jcts = []
        makespans = []
        rewards = []
        for j in range(NUM_TEST):
            dir_path = root_dir + exp_name + "_" + str(test_value) + "/" + str(j) + '/'
            file = dir_path + exp_name + "_" + str(test_value) + "/rl_validation.txt"
            assert os.path.exists(file)
            f = open(file, 'r')
            temp_jcts = []
            temp_makespans = []
            temp_rewards = []
            for line in f:
                tmp = line.replace("\n", '').split(" ")
                temp_jcts.append(float(tmp[2]))
                temp_makespans.append(float(tmp[3]))
                temp_rewards.append(float(tmp[4]))
            # find the min jct
            min_index = np.argmin(temp_jcts)
            jcts.append(temp_jcts[min_index])
            makespans.append(temp_makespans[min_index])
            rewards.append(temp_rewards[min_index])
        results[test_value] = (str(np.average(jcts)) + "+-" + str(np.std(jcts)),
                               str(np.average(makespans)) + "+-" + str(np.std(makespans)),
                               str(np.average(rewards)) + "+-" + str(np.std(rewards)))
    f = open(root_dir + "results.txt", "w")
    for item in results.items():
        f.write(str(item) + "\n")
    f.close()
    print(results)
    return results


def _sl_rl(dir_path, config, device):
    """
    Start the SL and RL training.
    """
    # SL
    sl_config = copy.deepcopy(sl_cfg)
    for key, value in config.items():
        if key not in sl_config:  # sl_cfg has higher priority
            sl_config[key] = value
    os.system("mkdir -p " + dir_path)
    os.system("cp *.py *.txt " + dir_path)
    replace_params(sl_config, dir_path)
    if TASK_ID != 17:
        os.system("cd " + dir_path + " && CUDA_VISIBLE_DEVICES=" + str(device) + " python train.py")
    else:
        os.system("cd " + dir_path + " && python train.py")

    time.sleep(3)

    # RL
    replace_params(config, dir_path)
    if TASK_ID != 17:
        os.system("cd " + dir_path + " && CUDA_VISIBLE_DEVICES=" + str(device) + " python train.py")
    else:
        os.system("cd " + dir_path + " && python train.py")


def _baseline(dir_path, config):
    """
    Start the baselines.
    """
    os.system("mkdir -p " + dir_path)
    os.system("cp *.py *.txt " + dir_path)
    replace_params(config, dir_path)
    os.system("cd " + dir_path + " && python comparison.py")


def run(id, exp_name, test_vals):
    print("Running experiments for", exp_name)
    tic = time.time()
    root_dir = exp_name + "-" + datetime.datetime.today().strftime("%Y%m%d_%H%M%S") + "/"

    pool = multiprocessing.Pool(processes=PARALLELISM)
    for i in range(len(test_vals)):
        test_val = test_vals[i]
        print("Testing", exp_name, "with value", test_val)
        parent_dir = root_dir + exp_name + "_" + str(test_val) + "/"
        for j in range(NUM_TEST):
            print("round", j)
            dir_path = parent_dir + str(j) + "/"
            cfg = get_cfg(id, exp_name, test_val)
            device = (i * NUM_TEST + j) % 2
            if id in [12, 13, 14, 15]:
                pool.apply_async(_baseline, args=(dir_path, cfg))
            else:
                pool.apply_async(_sl_rl, args=(dir_path, cfg, device))
            if id in [12, 13, 14, 15]:
                time.sleep(0.3)
            else:
                time.sleep(3)

    pool.close()
    pool.join()

    results = process_results(root_dir, exp_name, test_vals)
    print("Finish testing all values of", exp_name)
    print("The result is:", results)
    toc = time.time()
    print("Elapsed time:", toc - tic, "seconds")


def main(id):
    global PARALLELISM, TASK_ID
    TASK_ID = id

    exp_name = ""
    test_values = None
    if id == 1:
        exp_name = "sched_window_size"
        test_values = [10, 20, 30, 40, 50, 60]
    elif id == 2:
        exp_name = "number_of_neurons"
        test_values = [16, 32, 64, 96, 128, 160, 192, 256]
    elif id == 3:
        PARALLELISM = 5
        exp_name = "number_of_hidden_layers"
        test_values = [1, 2, 3, 4]
    elif id == 4:
        exp_name = "bundle_action"  # bundle false error
        test_values = [False, True]
    elif id == 5:
        exp_name = "job_arrival_distribution"
        test_values = ["Ali_Trace", "Uniform", "Poisson", "Google_Trace"]
    elif id == 6:
        exp_name = "batch_normalization"
        test_values = [False, True]
    elif id == 7:
        exp_name = "sl_loss_function"
        test_values = ["Mean_Square", "Cross_Entropy", "Absolute_Difference"]
    elif id == 8:
        exp_name = "job_reward_function"
        test_values = ["Norm_Progress", "Job_Progress", "Num_Uncompleted_Jobs"]
    elif id == 9:
        exp_name = "experience_replay"
        test_values = [False, True]
    elif id == 10:
        exp_name = "critic_network"
        test_values = [False, True]
    elif id == 11:
        exp_name = "exploration"
        test_values = [False, True]
    elif id == 12:
        exp_name = "DRF_baseline"
        test_values = ["Ali_Trace", "Uniform", "Poisson", "Google_Trace"]
    elif id == 13:
        exp_name = "SRTF_baseline"
        test_values = ["Ali_Trace", "Uniform", "Poisson", "Google_Trace"]
    elif id == 14:
        exp_name = "Tetris_baseline"
        test_values = ["Ali_Trace", "Uniform", "Poisson", "Google_Trace"]
    elif id == 15:
        exp_name = "Optimus_baseline"
        test_values = ["Ali_Trace", "Uniform", "Poisson", "Google_Trace"]
    elif id == 16:
        exp_name = "SL_heuristics"
        test_values = ["Optimus", "FIFO", "SRTF"]
    elif id == 17:
        PARALLELISM = 5
        exp_name = "a3c"
        test_values = [5, 4, 3, 2, 1]
    elif id == 18:
        exp_name = "changing_job_types"
        test_values = [True]
    elif id == 19:
        exp_name = "analytical_model"
        test_values = [False]
    elif id == 20:
        exp_name = "cluster_scale"
        test_values = ["large-4", "large-5", "large-6", "large-1", "large-2", "large-3", "testbed", "small"]
    elif id == 21:
        exp_name = "job_resr_balance"
        test_values = [True, False]
    elif id == 22:
        exp_name = "enable_SL_or_not"
        test_values = [True, False]
    elif id == 23:
        exp_name = "estimation_error_num_epoch"  # error
        test_values = [0.05, 0.1, 0.15, 0.2, 0.25]
    elif id == 24:
        PARALLELISM = 3
        exp_name = "number_of_hidden_layers"
        test_values = [5, 6, 7]
    elif id == 25:
        exp_name = "train_speed_error"
        test_values = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

    run(id, exp_name, test_values)


if len(sys.argv) != 2:
    print("a script for running experiment")
    print("Usage: please input one of following experiment IDs")
    print("1: scheduling window size")
    print("2: number of neurons")
    print("3: number of hidden layers")
    print("4: bundle action")
    print("5: job arrival distribution")
    print("6: batch normalization")
    print("7: sl loss function")
    print("8: job reward function")
    print("9: experience replay")
    print("10: critic network")
    print("11: exploration")
    print("12: DRF baseline")
    print("13: SRTF baseline")
    print("14: Tetris baseline")
    print("15: Optimus baseline")
    print("16: SL heuristics")
    print("17: a3c, change train_a3c.py to train.py, change parallelism, "
          "make sure a correct total batch size before running")
    print("18: changing job types during training")
    print("19: training on analytical model")
    print("20: cluster scale")
    print("21: job resource balance")
    print("22: enable SL or not")
    print("23: estimation error of epoch number")
    print("24: number of hidden layers")
    print("25: train speed error")
    exit(1)

main(int(sys.argv[1]))
