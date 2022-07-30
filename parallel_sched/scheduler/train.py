"""
The training module.
"""
import time
import numpy as np
import multiprocessing
import tensorflow as tf
import os
import parameter as param
import trace
from scheduler.dl2 import network
from scheduler.dl2 import dl2
from scheduler.drf import drf
from scheduler.fifo import fifo
from scheduler.srtf import srtf
from scheduler.tetris import tetris
from scheduler.optimus import optimus
import log
import validate
import memory
import prioritized_memory
import tb_log
import copy
import comparison

LOG_DIR = ""


def log_config(tb_logger):
    """
    Log the config of the supervised model (sl) and the reinforcement learning model (rl) to files.
    """
    global LOG_DIR
    if param.EXPERIMENT_NAME is None:
        LOG_DIR = "./backup/"
    else:
        LOG_DIR = "./" + param.EXPERIMENT_NAME + "/"

    os.system("mkdir -p " + LOG_DIR + "; cp *.py *.txt " + LOG_DIR)

    pm_md = globals().get("pt", None)
    train_config = dict()
    if pm_md:
        train_config = {
            key: val for key, val in pm_md.__dict__.items() if not (key.startswith("__") or key.startswith("_"))
        }
    train_config_str = ""
    for key, val in train_config.items():
        train_config_str += "{:<30}{:<100}".format(key, val) + "\n\n"

    tb_logger.add_text(tag="Config", value=train_config_str, step=0)
    tb_logger.flush()

    if param.TRAINING_MODE == "SL":
        f = open(param.MODEL_DIR + "sl_model.config", "w")
    else:
        f = open(param.MODEL_DIR + "rl_model.config", "w")
    f.write(train_config_str)
    f.close()

    f = open(LOG_DIR + "config.md", 'w')
    f.write(train_config_str)
    f.close()


def collect_stats(stats_qs, tb_logger, step):
    """
    Collect the statistics from stats_qs through the logger.
    """
    policy_entropys = []
    policy_losses = []
    value_losses = []
    td_losses = []
    step_rewards = []
    jcts = []
    makespans = []
    rewards = []
    val_losses = []
    val_jcts = []
    val_makespans = []
    val_rewards = []
    for id in range(param.NUM_AGENTS):
        while not stats_qs[id].empty():
            stats = stats_qs[id].get()
            tag_prefix = "SAgent " + str(id) + " "
            if stats[0] == "step:sl":
                _, entropy, loss = stats
                policy_entropys.append(entropy)
                policy_losses.append(loss)
                if id < param.NUM_RECORD_AGENTS and param.EXPERIMENT_NAME is None:
                    tb_logger.add_scalar(tag=tag_prefix + "SL Loss", value=loss, step=step)
                    tb_logger.add_scalar(tag=tag_prefix + "SL Entropy", value=entropy, step=step)

            elif stats[0] == "val":
                _, val_loss, jct, makespan, reward = stats
                val_losses.append(val_loss)
                val_jcts.append(jct)
                val_makespans.append(makespan)
                val_rewards.append(reward)
                if id < param.NUM_RECORD_AGENTS and param.EXPERIMENT_NAME is None:
                    tb_logger.add_scalar(tag=tag_prefix + "Val Loss", value=val_loss, step=step)
                    tb_logger.add_scalar(tag=tag_prefix + "Val JCT", value=jct, step=step)
                    tb_logger.add_scalar(tag=tag_prefix + "Val Makespan", value=makespan, step=step)
                    tb_logger.add_scalar(tag=tag_prefix + "Val Reward", value=reward, step=step)

            elif stats[0] == "step:policy":
                _, entropy, loss, td_loss, step_reward, output = stats
                policy_entropys.append(entropy)
                policy_losses.append(loss)
                td_losses.append(td_loss)
                step_rewards.append(step_reward)
                if id < param.NUM_RECORD_AGENTS and param.EXPERIMENT_NAME is None:
                    tb_logger.add_scalar(tag=tag_prefix + "Policy Entropy", value=entropy, step=step)
                    tb_logger.add_scalar(tag=tag_prefix + "Policy Loss", value=loss, step=step)
                    tb_logger.add_scalar(tag=tag_prefix + "TD Loss", value=td_loss, step=step)
                    tb_logger.add_scalar(tag=tag_prefix + "Step Reward", value=step_reward, step=step)
                    tb_logger.add_histogram(tag=tag_prefix + "Output", value=output, step=step)

            elif stats[0] == "step:policy+value":
                _, entropy, policy_loss, value_loss, td_loss, step_reward, output = stats
                policy_entropys.append(entropy)
                policy_losses.append(policy_loss)
                value_losses.append(value_loss)
                td_losses.append(td_loss)
                step_rewards.append(step_reward)
                if id < param.NUM_RECORD_AGENTS and param.EXPERIMENT_NAME is None:
                    tb_logger.add_scalar(tag=tag_prefix + "Policy Entropy", value=entropy, step=step)
                    tb_logger.add_scalar(tag=tag_prefix + "Policy Loss", value=policy_loss, step=step)
                    tb_logger.add_scalar(tag=tag_prefix + "Value Loss", value=value_loss, step=step)
                    tb_logger.add_scalar(tag=tag_prefix + "TD Loss", value=td_loss, step=step)
                    tb_logger.add_scalar(tag=tag_prefix + "Step Reward", value=step_reward, step=step)
                    tb_logger.add_histogram(tag=tag_prefix + "Output", value=output, step=step)

            elif stats[0] == "trace:sched_result":
                _, jct, makespan, reward = stats
                jcts.append(jct)
                makespans.append(makespan)
                rewards.append(reward)
                if id < param.NUM_RECORD_AGENTS and param.EXPERIMENT_NAME is None:
                    tb_logger.add_scalar(tag=tag_prefix + "Avg JCT", value=jct, step=step)
                    tb_logger.add_scalar(tag=tag_prefix + "Makespan", value=makespan, step=step)
                    tb_logger.add_scalar(tag=tag_prefix + "Reward", value=reward, step=step)

            elif stats[0] == "trace:job_stats":
                _, episode, jobstats = stats
                if id < param.NUM_RECORD_AGENTS and param.EXPERIMENT_NAME is None:
                    job_stats_tag_prefix = tag_prefix + "Trace " + str(episode) + " Step " + str(step) + " "
                    for i in range(len(jobstats["arrival"])):
                        tb_logger.add_scalar(tag=job_stats_tag_prefix + "Arrival", value=jobstats["arrival"][i], step=i)
                    for i in range(len(jobstats["ts_completed"])):
                        tb_logger.add_scalar(tag=job_stats_tag_prefix + "Ts_completed",
                                             value=jobstats["ts_completed"][i], step=i)
                    for i in range(len(jobstats["tot_completed"])):
                        tb_logger.add_scalar(tag=job_stats_tag_prefix + "Tot_completed",
                                             value=jobstats["tot_completed"][i], step=i)
                    for i in range(len(jobstats["uncompleted"])):
                        tb_logger.add_scalar(tag=job_stats_tag_prefix + "Uncompleted", value=jobstats["uncompleted"][i],
                                             step=i)
                    for i in range(len(jobstats["running"])):
                        tb_logger.add_scalar(tag=job_stats_tag_prefix + "Running", value=jobstats["running"][i], step=i)
                    for i in range(len(jobstats["total"])):
                        tb_logger.add_scalar(tag=job_stats_tag_prefix + "Total jobs", value=jobstats["total"][i],
                                             step=i)
                    for i in range(len(jobstats["backlog"])):
                        tb_logger.add_scalar(tag=job_stats_tag_prefix + "Backlog", value=jobstats["backlog"][i], step=i)
                    for i in range(len(jobstats["cpu_util"])):
                        tb_logger.add_scalar(tag=job_stats_tag_prefix + "CPU_Util", value=jobstats["cpu_util"][i],
                                             step=i)
                    for i in range(len(jobstats["gpu_util"])):
                        tb_logger.add_scalar(tag=job_stats_tag_prefix + "GPU_Util", value=jobstats["gpu_util"][i],
                                             step=i)
                    tb_logger.add_histogram(tag=job_stats_tag_prefix + "JCT", value=jobstats["duration"], step=step)

    tag_prefix = "Central "
    if len(policy_entropys) > 0:
        tb_logger.add_scalar(tag=tag_prefix + "Policy Entropy", value=sum(policy_entropys) / len(policy_entropys),
                             step=step)
    if len(policy_losses) > 0:
        tb_logger.add_scalar(tag=tag_prefix + "Policy Loss", value=sum(policy_losses) / len(policy_losses), step=step)
    if len(value_losses) > 0:
        tb_logger.add_scalar(tag=tag_prefix + "Value Loss", value=sum(value_losses) / len(value_losses), step=step)
    if len(td_losses) > 0:
        tb_logger.add_scalar(tag=tag_prefix + "TD Loss / Advantage", value=sum(td_losses) / len(td_losses), step=step)
    if len(step_rewards) > 0:
        tb_logger.add_scalar(tag=tag_prefix + "Batch Reward", value=sum(step_rewards) / len(step_rewards), step=step)
    if len(jcts) > 0:
        tb_logger.add_scalar(tag=tag_prefix + "JCT", value=sum(jcts) / len(jcts), step=step)
        # log results
        if param.TRAINING_MODE == "SL":
            f = open(LOG_DIR + "sl_train_jct.txt", 'a')
        else:
            f = open(LOG_DIR + "rl_train_jct.txt", 'a')
        f.write("step " + str(step) + ": " + str(sum(jcts) / len(jcts)) + "\n")
        f.close()
    if len(makespans) > 0:
        tb_logger.add_scalar(tag=tag_prefix + "Makespan", value=sum(makespans) / len(makespans), step=step)
        # log results
        if param.TRAINING_MODE == "SL":
            f = open(LOG_DIR + "sl_train_makespan.txt", 'a')
        else:
            f = open(LOG_DIR + "rl_train_makespan.txt", 'a')
        f.write("step " + str(step) + ": " + str(sum(makespans) / len(makespans)) + "\n")
        f.close()
    if len(rewards) > 0:
        tb_logger.add_scalar(tag=tag_prefix + "Reward", value=sum(rewards) / len(rewards), step=step)
    if len(val_losses) > 0:
        tb_logger.add_scalar(tag=tag_prefix + "Val Loss", value=sum(val_losses) / len(val_losses), step=step)
    if len(val_jcts) > 0:
        tb_logger.add_scalar(tag=tag_prefix + "Val JCT", value=sum(val_jcts) / len(val_jcts), step=step)
    if len(val_makespans) > 0:
        tb_logger.add_scalar(tag=tag_prefix + "Val Makespan", value=sum(val_makespans) / len(val_makespans), step=step)
    if len(val_rewards) > 0:
        tb_logger.add_scalar(tag=tag_prefix + "Val Reward", value=sum(val_rewards) / len(val_rewards), step=step)

    tb_logger.flush()


def test(policy_net, validation_traces, logger, step, tb_logger):
    val_tic = time.time()
    tag_prefix = "Central "

    try:
        if param.TRAINING_MODE == "SL":
            val_loss = validate.val_loss(policy_net, copy.deepcopy(validation_traces), logger, step)
            tb_logger.add_scalar(tag=tag_prefix + "Val Loss", value=val_loss, step=step)
        jct, makespan, reward = validate.val_jmr(policy_net, copy.deepcopy(validation_traces), logger, step, tb_logger)

        tb_logger.add_scalar(tag=tag_prefix + "Val JCT", value=jct, step=step)
        tb_logger.add_scalar(tag=tag_prefix + "Val Makespan", value=makespan, step=step)
        tb_logger.add_scalar(tag=tag_prefix + "Val Reward", value=reward, step=step)
        tb_logger.flush()
        val_toc = time.time()
        logger.info("Central Agent:" + " Validation at step " + str(step) + " Time: " + '%.3f' % (val_toc - val_tic))

        # log results
        if param.TRAINING_MODE == "SL":
            f = open(LOG_DIR + "sl_validation.txt", 'a')
        else:
            f = open(LOG_DIR + "rl_validation.txt", 'a')
        f.write("step " + str(step) + ": " + str(jct) + " " + str(makespan) + " " + str(reward) + "\n")
        f.close()

        return jct, makespan, reward

    except Exception as e:
        logger.error("Error when validation! " + str(e))
        tb_logger.add_text(tag="validation error", value=str(e), step=step)


def central_agent(net_weights_qs, net_gradients_qs, stats_qs):
    logger = log.get_logger(name="central_agent", level=param.LOG_MODE)
    logger.info("Start central agent...")

    if not param.RANDOMNESS:
        np.random.seed(param.np_seed)
        tf.set_random_seed(param.tf_seed)

    config = tf.ConfigProto()
    config.allow_soft_placement = False
    config.gpu_options.allow_growth = True
    tb_logger = tb_log.Logger(param.SUMMARY_DIR)
    log_config(tb_logger)

    with tf.Session(config=config) as sess:
        policy_net = network.PolicyNetwork(sess, "policy_net", param.TRAINING_MODE, logger)
        if param.VALUE_NET:
            value_net = network.ValueNetwork(sess, "value_net", param.TRAINING_MODE, logger)
        logger.info("Create the policy network with " + str(policy_net.get_num_weights()) + " parameters")

        sess.run(tf.global_variables_initializer())
        tb_logger.add_graph(sess.graph)
        tb_logger.flush()

        policy_tf_saver = tf.train.Saver(max_to_keep=param.MAX_NUM_CHECKPOINTS,
                                         var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="policy_net"))
        if param.POLICY_NN_MODEL is not None:
            policy_tf_saver.restore(sess, param.POLICY_NN_MODEL)
            logger.info("Policy model " + param.POLICY_NN_MODEL + " is restored.")

        if param.VALUE_NET:
            value_tf_saver = tf.train.Saver(max_to_keep=param.MAX_NUM_CHECKPOINTS,
                                            var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                                       scope='value_net'))
            if param.VALUE_NN_MODEL is not None:
                value_tf_saver.restore(sess, param.VALUE_NN_MODEL)
                logger.info("Value model " + param.VALUE_NN_MODEL + " is restored.")

            step = 1
            start_t = time.time()

            if param.VAL_ON_MASTER:
                validation_traces = []
                tags_prefix = ["DRF: ", "SRTF: ", "FIFO: ", "Tetris: ", "Optimus: "]
                for i in range(param.VAL_DATASET):
                    validation_traces.append(trace.Trace(None).get_trace())
                # TODO: here


def sl_agent(net_weights_qs, net_gradients_qs, stats_qs, agent_id):
    pass


def rl_agent(net_weights_qs, net_gradients_qs, stats_qs, agent_id):
    pass


def main():
    os.system("rm -f *.log")
    os.system("sudo kill -9 tensorboard; sleep 3")

    net_weights_qs = [multiprocessing.Queue(1)] * param.NUM_AGENTS
    net_gradients_qs = [multiprocessing.Queue(1)] * param.NUM_AGENTS
    stats_qs = [multiprocessing.Queue()] * param.NUM_AGENTS

    os.system("mkdir -p " + param.MODEL_DIR + "; mkdir -p " + param.SUMMARY_DIR)
    if param.EXPERIMENT_NAME is None:
        cmd = "cd " + param.SUMMARY_DIR + " && rm -rf *; tensorboard --logdir=./"
        board = multiprocessing.Process(target=lambda: os.system(cmd), args=())
        board.start()
        time.sleep(3)

    master = multiprocessing.Process(target=central_agent, args=(net_weights_qs, net_gradients_qs, stats_qs,))
    master.start()

    agents = None
    if param.TRAINING_MODE == "SL":
        agents = [multiprocessing.Process(
            target=sl_agent, args=(net_weights_qs[i], net_gradients_qs[i], stats_qs[i], i)
        ) for i in range(param.NUM_AGENTS)]

    elif param.TRAINING_MODE == "RL":
        agents = [multiprocessing.Process(
            target=rl_agent, args=(net_weights_qs[i], net_gradients_qs[i], stats_qs[i], i)
        ) for i in range(param.NUM_AGENTS)]

    for i in range(param.NUM_AGENTS):
        agents[i].start()

    master.join()


main()
