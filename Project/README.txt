
1) install package by running:

$ python setup.py develop

##############################################
##############################################

2)install other dependencies

install dependencies locally, by running:
$ pip install -r requirements.txt

##############################################
##############################################

3) code:

Blanks to be filled in are marked with "TODO"
The following files have TODOs in them:
- scripts/run_project_behavior_cloning.py
- infrastructure/rl_trainer.py
- infrastructure/torch_utils.py
- agents/bc_agent.py
- policies/MLP_policy.py
- policies/loaded_gaussian_policy.py

See the code + the hw pdf for more details.

##############################################
##############################################

4) run code: 

Run the following command for Imitation Learning:

$python project/scripts/run_project_behavior_cloning.py --expert_policy_file project/models/CartPole-v0.tar --env_name CartPole-v0 --exp_name test_bc_Cart --n_iter 1

$python project/scripts/run_project_behavior_cloning.py --expert_policy_file project/models/LunarLander-v2.tar --env_name LunarLander-v2 --exp_name test_bc_Lunar --n_iter 1

$python project/scripts/run_project_behavior_cloning.py --expert_policy_file project/models/LunarLanderContinuous-v2.tar --env_name LunarLanderContinuous-v2 --exp_name test_bc_LunarCont --n_iter 1

Run the following command for DAGGER:

$python project/scripts/run_project_behavior_cloning.py --expert_policy_file project/models/LunarLander-v2.tar --env_name LunarLander-v2 --exp_name test_dagger_Lunar --n_iter 10 --do_dagger

$python project/scripts/run_project_behavior_cloning.py --expert_policy_file project/models/LunarLanderContinuous-v2.tar --env_name LunarLanderContinuous-v2 --exp_name test_dagger_LunarCont --n_iter 10 --do_dagger

(NOTE: the --do_dagger flag, and the higher value for n_iter)

##############################################
##############################################

5) visualize saved tensorboard event file:

$ cd project/data/<your_log_dir>
$ tensorboard --logdir .

Then, navigate to shown url to see scalar summaries as plots (in 'scalar' tab), as well as videos (in 'images' tab)
