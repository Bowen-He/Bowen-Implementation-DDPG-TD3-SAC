Run\
  python experiment.py --hyper_param_name {Ant, BipedalWalker, HalfCheetah, Hopper, Humanoid, Pendulum, Walker} --experiment_name THE_EXPERIMENT_NAME --run_title THE_TRIAL_NAME --model {DDPG, TD3, SAC} --seed RANDOM_SEED\
to run experiments on gym MuJoCo.

Run\
  python plot_learning_curves.py\
to visualize the training record.
