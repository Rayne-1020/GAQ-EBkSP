"""Contains an experiment class for running simulations."""
from flow.core.util import emission_to_csv
from flow.utils.registry import make_create_env
from datetime import datetime
import logging
import time
import os
import numpy as np
import json



class Experiment:
    """
    Class for systematically running simulations in any supported simulator.

    This class acts as a runner for a network and environment. In order to use
    it to run an network and environment in the absence of a method specifying
    the actions of RL agents in the network, type the following:

        >>> from flow.envs import Env
        >>> flow_params = dict(...)  # see the examples in exp_config
        >>> exp = Experiment(flow_params)  # for some experiment configuration
        >>> exp.run(num_runs=1)

    If you wish to specify the actions of RL agents in the network, this may be
    done as follows:

        >>> rl_actions = lambda state: 0  # replace with something appropriate
        >>> exp.run(num_runs=1, rl_actions=rl_actions)

    Finally, if you would like to like to plot and visualize your results, this
    class can generate csv files from emission files produced by sumo. These
    files will contain the speeds, positions, edges, etc... of every vehicle
    in the network at every time step.

    In order to ensure that the simulator constructs an emission file, set the
    ``emission_path`` attribute in ``SimParams`` to some path.

        >>> from flow.core.params import SimParams
        >>> flow_params['sim'] = SimParams(emission_path="./data")

    Once you have included this in your environment, run your Experiment object
    as follows:

        >>> exp.run(num_runs=1, convert_to_csv=True)

    After the experiment is complete, look at the "./data" directory. There
    will be two files, one with the suffix .xml and another with the suffix
    .csv. The latter should be easily interpretable from any csv reader (e.g.
    Excel), and can be parsed using tools such as numpy and pandas.

    Attributes
    ----------
    custom_callables : dict < str, lambda >
        strings and lambda functions corresponding to some information we want
        to extract from the environment. The lambda will be called at each step
        to extract information from the env and it will be stored in a dict
        keyed by the str.
    env : flow.envs.Env
        the environment object the simulator will run
    """

    def __init__(self, flow_params, custom_callables=None):
        """Instantiate the Experiment class.

        Parameters
        ----------
        flow_params : dict
            flow-specific parameters
        custom_callables : dict < str, lambda >
            strings and lambda functions corresponding to some information we
            want to extract from the environment. The lambda will be called at
            each step to extract information from the env and it will be stored
            in a dict keyed by the str.
        """
        self.custom_callables = custom_callables or {}

        # Get the env name and a creator for the environment.
        create_env, _ = make_create_env(flow_params)

        # Create the environment.
        self.env = create_env()

        logging.info(" Starting experiment {} at {}".format(
            self.env.network.name, str(datetime.utcnow())))

        logging.info("Initializing environment.")

    def run(self, training, num_runs, model, debug):# values 从main 函数传进来的
        
        if debug:
            nb_steps_warmup = 2000#in the warmup steps, what is happen and what is not.
            batch_size = 32# what is batch size for?
            total_steps = 10000
            log_interval = 200#what is log_interval
            nb_max_episode_steps = 5
            gamma = 0.99 #discount factor in rl
        else:
            nb_steps_warmup = 2000
            batch_size = 32
            total_steps = 8000
            log_interval = 200
            nb_max_episode_steps = 10
            gamma = 0.99 #discount factor in rl
        
        F = 2 #features number: velocity and V/C
        N = 6 #total number of Fog Nodes
        A = 5 #action number: [1,2,3,4,5]

        from gym.spaces.box import Box 
        from gym.spaces import Discrete
        from gym.spaces.dict import Dict 

        states = Box(low = -np.inf, high = np.inf, shape = (N,F), dtype = np.float32)
        adjacency = Box(low = 0, high = 1, shape = (N,N), dtype = np.int32)
        #mask = Box(low = 0, high = 1, shape = (N, ), dtype = np.int32) GAT research 用不到这个

        obs_space = Dict({'states':states, 'adjacency':adjacency})  # no 'mask'
        act_space = Box(low = 0, high = 1, shape = (N, ), dtype = np.int32)#和mask看起来一毛一样

        from rl_gat_model import GATQNetwork_tf2 #from the GAT file import the GAT model
        from rl_gat_model import GraphicQNetwork #from the GAT file import the GraphConv model as the baseline model
        from rl_agents.memory import SequentialMemory
        from rl_agents.processor import Jiqian_MultiInputProcessor
        #from rl_agents.dqn import DQNAgent # The rl_agent DQN used here
        from rl_agents.dqn_200 import DQNAgent
        from rl_agents.policy import eps_greedy_q_policy, greedy_q_policy, random_obs_policy
        from spektral.layers import GATConv #the GAT model from the skeptral
        from spektral.layers import GCNConv #the GCN model from the skeptral
        from tensorflow.keras.optimizers import Adam
        import tensorflow as tf 


        memory_buffer = SequentialMemory(limit = 100000, window_length = 1)
        multi_input_processor = Jiqian_MultiInputProcessor(2)#my input: features and adjacency matrix

        if model == 'gat':
            print("yes")
            rl_model = GATQNetwork_tf2(N, F)
        if model == 'gcn':
            rl_model = GraphicQNetwork(N, F)
        # else:
        #     raise NotImplementedError

        my_dqn = DQNAgent(processor = multi_input_processor,
                          model = rl_model.base_model,
                          policy = eps_greedy_q_policy(0.0),
                          test_policy = greedy_q_policy(),
                          start_policy = random_obs_policy(),
                          #nb_total_agents = N,
                          nb_actions = A,
                          nb_steps_warmup = nb_steps_warmup,
                          memory = memory_buffer,
                          batch_size = batch_size,
                          gamma = gamma,
                          custom_model_objects={'GATConv':GATConv})
                          #custom_model_objects = {'GCNConv':GCNConv})
        
        my_dqn.compile(Adam(0.0001))#learning rate

        if training:

            logdir = "./logs/"
            history_file = "./logs/_training_hist_GCN_max_step_10_near_num_15.txt"
            try:
                os.remove(history_file)
            except:
                pass

            from rl_agents.rl_lib.callbacks import FileLogger

            file_log = FileLogger(history_file)
            history = my_dqn.fit(self.env, action_repetition = 200, nb_steps = total_steps, nb_max_episode_steps = nb_max_episode_steps,
                visualize = False, verbose = 1, log_interval = log_interval, callbacks = [file_log])
            my_dqn.save_weights('./models/dqn_{}.GCN_max_step_10_near_num_15'.format('gat'), overwrite = True)#format


            #from generate_training_plots import plot_training

            #plot_training(logdir)

        else:
            history_file = "./logs/test/{}_test_hist_2_change_road_index.txt".format('gat')
            my_dqn.load_weights('./models/dqn_{}.2_change_road_index'.format('gat'))
            print("successfully loaded")
            hist = my_dqn.test(self.env, nb_episodes = num_runs, action_repetition = 200, nb_max_episode_steps = nb_max_episode_steps)

            with open(histoey_file,'w') as f:
                json.dump(hist.history, f)

        
        







