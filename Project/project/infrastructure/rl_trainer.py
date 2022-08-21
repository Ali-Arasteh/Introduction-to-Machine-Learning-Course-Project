import time
from collections import OrderedDict
import pickle
import numpy as np
import torch
import gym
import os
from project.infrastructure.utils import *
from project.infrastructure.logger import Logger

# params for saving rollout videos to tensorboard
MAX_NVIDEO = 2
MAX_VIDEO_LEN = 40

class RL_Trainer(object):
    def __init__(self, params):
        #############
        ## INIT
        #############
        # Get params, create logger, create TF session
        self.params = params
        self.logger = Logger(self.params['logdir'])
        # Set random seeds
        seed = self.params['seed']
        torch.manual_seed(seed)
        np.random.seed(seed)
        #############
        ## ENV
        #############
        # Make the gym environment
        self.env = gym.make(self.params['env_name'])
        self.env.seed(seed)
        # Maximum length for episodes
        self.params['ep_len'] = self.params['ep_len'] or self.env.spec.max_episode_steps
        print(self.params['ep_len'])
        # Is this env continuous, or self.discrete?
        discrete = isinstance(self.env.action_space, gym.spaces.Discrete)
        self.params['agent_params']['discrete'] = discrete
        # Observation and action sizes
        ob_dim = self.env.observation_space.shape[0]
        ac_dim = self.env.action_space.n if discrete else self.env.action_space.shape[0]
        self.params['agent_params']['ac_dim'] = ac_dim
        self.params['agent_params']['ob_dim'] = ob_dim
        # simulation timestep, will be used for video saving
        if 'model' in dir(self.env):
            self.fps = 1/self.env.model.opt.timestep
        else:
            self.fps = self.env.env.metadata['video.frames_per_second']
        #############
        ## AGENT
        #############
        agent_class = self.params['agent_class']
        self.agent = agent_class(self.env, self.params['agent_params'])
        self.counter = 0

    def run_training_loop(self, n_iter, collect_policy, eval_policy,
                          expert_policy, relabel_with_expert=False,
                          start_relabel_with_expert=1):
        """
        :param n_iter:  number of (dagger) iterations
        :param collect_policy:
        :param eval_policy:
        :param expert_policy:
        :param relabel_with_expert:  whether to perform dagger
        :param start_relabel_with_expert: iteration at which to start relabel with expert
        """
        # init vars at beginning of training
        self.total_envsteps = 0
        self.start_time = time.time()
        for itr in range(n_iter):
            print("\n\n********** Iteration %i ************"%itr)
            # decide if videos should be rendered/logged at this iteration
            if itr % self.params['video_log_freq'] == 0 and self.params['video_log_freq'] != -1:
                self.log_video = True
            else:
                self.log_video = False
            # decide if metrics should be logged
            if itr % self.params['scalar_log_freq'] == 0 and self.params['scalar_log_freq'] != -1:
                self.log_metrics = True
            else:
                self.log_metrics = False
            # collect trajectories, to be used for training
            training_returns = self.collect_training_trajectories(itr, expert_policy, collect_policy, self.params['batch_size']) ## implement this function below
            paths, envsteps_this_batch, train_video_paths = training_returns
            self.total_envsteps += envsteps_this_batch
            # relabel the collected obs with actions from a provided expert policy
            if relabel_with_expert and itr>=start_relabel_with_expert:
                paths = self.do_relabel_with_expert(expert_policy, paths) ## implement this function below
            # add collected data to replay buffer
            self.agent.add_to_replay_buffer(paths)
            # train agent (using sampled data from replay buffer)
            loss = self.train_agent()
            print("itr #", itr, ":", "  loss:", loss)
            # log/save
            if self.log_video or self.log_metrics:
                # perform logging
                print('\nBeginning logging procedure...')
                self.perform_logging(itr, paths, eval_policy, train_video_paths)
                # save policy
                print('\nSaving agent\'s actor...')
                self.agent.actor.save(self.params['logdir'] + '/policy_itr_'+str(itr))

    def collect_training_trajectories(self, itr, expert_policy, collect_policy, batch_size):
        """
        :param itr:
        :param load_initial_expertdata:  path to expert data pkl file
        :param collect_policy:  the current policy using which we collect data
        :param batch_size:  the number of transitions we collect
        :return:
            paths: a list trajectories
            envsteps_this_batch: the sum over the numbers of environment steps in paths
            train_video_paths: paths which also contain videos for visualization purposes
        """
        # decide whether to load training data or use
        # HINT1: depending on if it's the first iteration or not,
            # decide whether to either
                # collect data via expert policy to be used as a dataset. In this case you can directly return as follows
                # ``` return loaded_paths, 0, None ```

                # collect data via collect_policy (which is out training policy), batch_size is the number of transitions you want to collect.
                # you want each of these collected rollouts to be of length self.params['ep_len']
        if itr == 0:
            print("\nCollecting data from the expert policy...")
            paths, _ = sample_trajectories(self.env, expert_policy, 2 * batch_size, self.params['ep_len'])
            return paths, 0, None
        else:
            print("\nCollecting data to be used for training...")
            paths, envsteps_this_batch = sample_trajectories(self.env, collect_policy, batch_size, self.params['ep_len'])
            # collect more rollouts with the same policy, to be saved as videos in tensorboard
            # note: here, we collect MAX_NVIDEO rollouts, each of length MAX_VIDEO_LEN
            train_video_paths = None
            if self.log_video:
                print('\nCollecting train rollouts to be used for saving videos...')
                train_video_paths = sample_n_trajectories(self.env, collect_policy, MAX_NVIDEO, MAX_VIDEO_LEN, True)
            return paths, envsteps_this_batch, train_video_paths

    def batching_data(self, paths, batch_size):
        batched_pathed = []
        for path in paths:
            obs = path["observation"]
            imobs = path["image_obs"]
            rews = path["reward"]
            acs = path["action"]
            nobs = path["next_observation"]
            ters = path["terminal"]
            n = obs.shape[0]
            indecis = np.random.choice(np.arange(n))[:batch_size]
            batched_pathed.append({"observation": obs[indecis], "image_obs": imobs[indecis], "reward": rews[indecis], "action": obs[acs], "next_observation": nobs[indecis], "terminal": ters[indecis]})
        return batched_pathed

    def train_agent(self):
        print('\nTraining agent using sampled data from replay buffer...')
        step = self.params['num_agent_train_steps_per_iter']//5
        for train_step in range(self.params['num_agent_train_steps_per_iter']):
            # sample some data from the data buffer
            ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch = self.agent.sample(self.params['train_batch_size'])
            # use the sampled data for training
            loss = self.agent.train(ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch)
            self.logger.log_scalar(loss, "losses", self.counter)
            self.counter += 1
            if train_step % step == 0:
                print("train step #", train_step, "  loss: ", loss)
        return loss

    def do_relabel_with_expert(self, expert_policy, paths):
        print("\nRelabelling collected observations with labels from an expert policy...")
        # relabel collected observations (from our policy) with labels from an expert policy
        # HINT: query the policy (using the get_action function) with paths[i]["observation"]
        # and replace paths[i]["action"] with these expert labels
        for path in paths:
            path["action"] = expert_policy.get_action(path["observation"])
        return paths

    def perform_logging(self, itr, paths, eval_policy, train_video_paths):
        # collect eval trajectories, for logging
        print("\nCollecting data for eval...")
        eval_paths, eval_envsteps_this_batch = sample_trajectories(self.env, eval_policy, self.params['eval_batch_size'], self.params['ep_len'])
        # save eval rollouts as videos in tensorboard event file
        if self.log_video and train_video_paths != None:
            print('\nCollecting video rollouts eval')
            eval_video_paths = sample_n_trajectories(self.env, eval_policy, MAX_NVIDEO, MAX_VIDEO_LEN, True)
            # save train/eval videos
            print('\nSaving train rollouts as videos...')
            self.logger.log_paths_as_videos(train_video_paths, itr, fps=self.fps, max_videos_to_save=MAX_NVIDEO, video_title='train_rollouts')
            self.logger.log_paths_as_videos(eval_video_paths, itr, fps=self.fps,max_videos_to_save=MAX_NVIDEO, video_title='eval_rollouts')

        # save eval metrics
        if self.log_metrics:
            # returns, for logging
            train_returns = [path["reward"].sum() for path in paths]
            eval_returns = [eval_path["reward"].sum() for eval_path in eval_paths]
            # episode lengths, for logging
            train_ep_lens = [len(path["reward"]) for path in paths]
            eval_ep_lens = [len(eval_path["reward"]) for eval_path in eval_paths]
            # decide what to log
            logs = OrderedDict()
            logs["Eval_AverageReturn"] = np.mean(eval_returns)
            logs["Eval_StdReturn"] = np.std(eval_returns)
            logs["Eval_MaxReturn"] = np.max(eval_returns)
            logs["Eval_MinReturn"] = np.min(eval_returns)
            logs["Eval_AverageEpLen"] = np.mean(eval_ep_lens)
            logs["Train_AverageReturn"] = np.mean(train_returns)
            logs["Train_StdReturn"] = np.std(train_returns)
            logs["Train_MaxReturn"] = np.max(train_returns)
            logs["Train_MinReturn"] = np.min(train_returns)
            logs["Train_AverageEpLen"] = np.mean(train_ep_lens)
            logs["Train_EnvstepsSoFar"] = self.total_envsteps
            logs["TimeSinceStart"] = time.time() - self.start_time
            if itr == 0:
                self.initial_return = np.mean(train_returns)
            logs["Initial_DataCollection_AverageReturn"] = self.initial_return
            # perform the logging
            for key, value in logs.items():
                print('{} : {}'.format(key, value))
                self.logger.log_scalar(value, key, itr)
            print('Done logging...\n\n')
            self.logger.flush()