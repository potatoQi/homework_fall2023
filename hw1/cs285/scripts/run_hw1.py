"""
Runs behavior cloning and DAgger for homework 1

Functions to edit:
    1. run_training_loop
"""

import pickle   # 用于保存和加载Python对象（如专家策略数据）
import os
import time
import gym

import numpy as np
import torch

# cs285.infrastructure是自定义库，其中包含了网络模型、工具函数、日志记录、回放缓冲区等功能
from cs285.infrastructure import pytorch_util as ptu                    # PyTorch使用函数
from cs285.infrastructure import utils                                  # 辅助工具函数
from cs285.infrastructure.logger import Logger                          # 用于记录日志
from cs285.infrastructure.replay_buffer import ReplayBuffer             # 用于存储与环境交互的数据
from cs285.policies.MLP_policy import MLPPolicySL                       # MLP模型
from cs285.policies.loaded_gaussian_policy import LoadedGaussianPolicy  # 加载专家策略，这里的策略是一个分布


# how many rollouts to save as videos to tensorboard

# 这是一个常量，指定了在每次训练过程中保存多少个视频（方便同一训练轮中不同轨迹的对比）
MAX_NVIDEO = 2

# 这是一个常量，表示每个视频的最大长度（以环境步骤为单位）
MAX_VIDEO_LEN = 40  # we overwrite this in the code below

# 这是一个包含多个字符串的列表，每个字符串都是一个环境的名称。这个列表列出了多个强化学习环境的名称
# 这些环境大多来自 OpenAI Gym 的 MuJoCo 仿真库，是强化学习中常用的基准环境，用于测试和评估算法。
# Ant-v4: 一个四足机器人环境，机器人可以在二维平面上行走，任务是学习如何走得更远
# Walker2d-v4: 一个双足机器人环境，任务是学习如何让机器人行走
# HalfCheetah-v4: 一个类人型机器人环境，任务是让机器人模拟类似猎豹的跑步动作
# Hopper-v4: 一个单足跳跃机器人环境，任务是学习如何在单足上跳跃
MJ_ENV_NAMES = ["Ant-v4", "Walker2d-v4", "HalfCheetah-v4", "Hopper-v4"]


def run_training_loop(params):
    """
    Runs training with the specified parameters
    (behavior cloning or dagger)
    传入的 params 是一个字典, 里面包含了训练的所有参数
    
    Args:
        params: experiment parameters
    """

    #############
    ## INIT
    #############

    # Get params, create logger, create TF session
    # 实例化了一个日志记录器, 相关文件将保存在 params['logdir'] 里
    logger = Logger(params['logdir'])

    # Set random seeds
    seed = params['seed']
    np.random.seed(seed)    # 设置了 numpy 里的随机种子
    torch.manual_seed(seed) # 设置了 PyTorch 里的随机种子
    # 这里的ptu是pytorch_util.py这个程序, 下面这句话意思是调用程序里的init_gpu函数
    ptu.init_gpu(
        use_gpu=not params['no_gpu'],
        gpu_id=params['which_gpu']
    )

    # Set logger attributes
    log_video = True    # 日志视频是否保存
    log_metrics = True  # 日志指标是否保存

    #############
    ## ENV
    #############

    # Make the gym environment
    env = gym.make(params['env_name'], render_mode=None)    # 创建一个名为 params['env_name'] 的环境, 不开启渲染模式
    env.reset(seed=seed)    # 初始化环境, 使用种子保证每次初始化完是一样的

    # Maximum length for episodes
    params['ep_len'] = params['ep_len'] or env.spec.max_episode_steps   # 如果 params['ep_len'] 为 None , 就用环境默认的最大步数
    MAX_VIDEO_LEN = params['ep_len']

    # gym 里的 actino_space 有两种类型：Discrete(离散)、Box(连续)
    assert isinstance(env.action_space, gym.spaces.Box), "Environment must be continuous"

    # Observation and action sizes
    # env.observation_space 代表环境的观测空间, 即 agent 能看到的信息范围
    # env.observation_space.shape[0] 代表了这个观测空间的观测维度。比如观测空间是个图像，那么观测的维度就是3，因为rgb有3层图像堆叠起来
    ob_dim = env.observation_space.shape[0]
    # env.action_space.shape[0] 代表了动作空间的动作数。比如agent可以控制5个关节，那么5就是action_space的维度
    ac_dim = env.action_space.shape[0]

    # simulation timestep, will be used for video saving
    # fps就是每秒的帧数
    if 'model' in dir(env): # 如果env里有model这个属性, 说明该环境中有一个引擎(例如MuJoCo物理引擎)来处理环境的状态更新
        fps = 1/env.model.opt.timestep
    else:
        fps = env.env.metadata['render_fps']

    #############
    ## AGENT
    #############

    # TODO: Implement missing functions in this class.
    # 实例化一个神经网络作为policy
    actor = MLPPolicySL(
        ac_dim, # 动作空间的维度(output)
        ob_dim, # 观测空间的维度(input)
        params['n_layers'],
        params['size'],
        learning_rate=params['learning_rate'],
    )

    # replay buffer
    # 实例化一个replay_buffer
    replay_buffer = ReplayBuffer(params['max_replay_buffer_size'])

    #######################
    ## LOAD EXPERT POLICY
    #######################

    # 专家策略加载到 expert_policy 中了
    print('Loading expert policy from...', params['expert_policy_file'])
    expert_policy = LoadedGaussianPolicy(params['expert_policy_file'])
    expert_policy.to(ptu.device)
    print('Done restoring expert policy...')

    #######################
    ## TRAINING LOOP
    #######################

    # init vars at beginning of training
    total_envsteps = 0
    start_time = time.time()

    # n_iter 是训练论述, 朴素BC就一轮, DAgger至少两轮
    for itr in range(params['n_iter']):
        print("\n\n********** Iteration %i ************"%itr)

        # decide if videos should be rendered/logged at this iteration
        # 是否要保存当轮训练的video
        log_video = ((itr % params['video_log_freq'] == 0) and (params['video_log_freq'] != -1))
        # decide if metrics should be logged
        # 是否要保存当轮训练的metrics
        log_metrics = (itr % params['scalar_log_freq'] == 0)

        print("\nCollecting data to be used for training...")
        if itr == 0:
            # BC training from expert data.
            # 第一次训练用的数据是专家数据
            paths = pickle.load(open(params['expert_data'], 'rb'))
            envsteps_this_batch = 0
        else:
            # DAGGER training from sampled data relabeled by expert
            assert params['do_dagger']
            # TODO: collect `params['batch_size']` transitions
            # HINT: use utils.sample_trajectories
            # TODO: implement missing parts of utils.sample_trajectory
            # paths, envsteps_this_batch = TODO

            # 收集到多条轨迹, 每次轨迹收集ep_len个数据, 直到收集到的步数达到batch_size
            paths, envsteps_this_batch = utils.sample_trajectories(env, actor, params['batch_size'], params['ep_len'])

            # relabel the collected obs with actions from a provided expert policy
            if params['do_dagger']:
                print("\nRelabelling collected observations with labels from an expert policy...")

                # TODO: relabel collected obsevations (from our policy) with labels from expert policy
                # HINT: query the policy (using the get_action function) with paths[i]["observation"]
                # and replace paths[i]["action"] with these expert labels
                # paths = TODO

                for i in range(len(paths)):
                    obs = paths[i]["observation"]
                    paths[i]["action"] = expert_policy.get_action(obs)

        total_envsteps += envsteps_this_batch
        # add collected data to replay buffer
        # 把收集到的轨迹数据都加入到replay_buffer里
        replay_buffer.add_rollouts(paths)

        # train agent (using sampled data from replay buffer)
        print('\nTraining agent using sampled data from replay buffer...')
        training_logs = []
        # 优化 num_agent_train_steps_per_iter 次
        for _ in range(params['num_agent_train_steps_per_iter']):

            # TODO: sample some data from replay_buffer
            # HINT1: how much data = params['train_batch_size']
            # HINT2: use np.random.permutation to sample random indices
            # HINT3: return corresponding data points from each array (i.e., not different indices from each array)
            # for imitation learning, we only need observations and actions.  
            # ob_batch, ac_batch = TODO
        
            # 打混采样 train_batch_size 个
            indices = np.random.permutation(len(replay_buffer))[:params['train_batch_size']]
            # 得到本次优化需要的 obs 和 acs
            ob_batch = torch.from_numpy(replay_buffer.obs[indices]).to(ptu.device)
            ac_batch = torch.from_numpy(replay_buffer.acs[indices]).to(ptu.device)

            # use the sampled data to train an agent
            train_log = actor.update(ob_batch, ac_batch)
            training_logs.append(train_log)

        # log/save
        print('\nBeginning logging procedure...')
        if log_video:
            # save eval rollouts as videos in tensorboard event file
            print('\nCollecting video rollouts eval')
            # 去利用当前policy采样 MAX_NVIDEO 条轨迹, 每条轨迹的长度是 MAX_VIDEO_LEN , 并且开启渲染模式
            eval_video_paths = utils.sample_n_trajectories(
                env, actor, MAX_NVIDEO, MAX_VIDEO_LEN, True
            )

            # save videos
            if eval_video_paths is not None:
                logger.log_paths_as_videos(
                    eval_video_paths, itr,
                    fps=fps,
                    max_videos_to_save=MAX_NVIDEO,
                    video_title='video'
                )

        if log_metrics:
            # save eval metrics
            print("\nCollecting data for eval...")

            # 每次采样 ep_len 长度的轨迹, 直到长度达到 eval_batch_size , 作为eval的轨迹
            eval_paths, eval_envsteps_this_batch = utils.sample_trajectories(
                env, actor, params['eval_batch_size'], params['ep_len'])

            # 计算当前训练集上的指标和评估集上的指标
            logs = utils.compute_metrics(paths, eval_paths)
            
            # compute additional metrics
            # 将最后一步梯度下降的训练日志包含进来
            logs.update(training_logs[-1]) # Only use the last log for now

            # 与环境交互了多少步(若干次训练的步数都包含)
            logs["Train_EnvstepsSoFar"] = total_envsteps
            
            logs["TimeSinceStart"] = time.time() - start_time
            if itr == 0:
                logs["Initial_DataCollection_AverageReturn"] = logs["Train_AverageReturn"]

            # perform the logging
            # 把此步日志保存到总日志logger中
            for key, value in logs.items():
                print('{} : {}'.format(key, value))
                logger.log_scalar(value, key, itr)
            print('Done logging...\n\n')

            logger.flush()

        # 保存当前policy的参数
        if params['save_params']:
            print('\nSaving agent params')
            actor.save('{}/policy_itr_{}.pt'.format(params['logdir'], itr))


def main():
    import argparse # 是Python标准库里的一个包, 专门用于处理命令行参数
    parser = argparse.ArgumentParser()  # 创建一个ArgumentParser对象, 用于解析和处理命令行输入的参数
    '''
    parser.add_argument('--长提示', '-短提示', ..., type=str/int/float, default=默认值, required=, action=, const=)

    --两个表示长提示, -一个表示短提示
    required如果是True, 但是你没输入这个参数的话, 就会报错
    default是如果你没输入这个参数, 那么程序就会使用默认值
    type是你输入的参数的类型, 如果跟type不符, 那么就会报错
    action表示当你输入参数的时候, 该如何处理该参数
        如果action='store', 默认
        如果action='store_const', 只要输了--参数, 那么就把const=...赋值给该参数, 否则赋值None
        如果action='store_true', 只要输了--参数, 那么就把True赋值给该参数, 否则赋值False
    '''

    # 这里是让你填专家策略文件的路径, 后缀名为pkl的文件通常是训练好的模型/数据/实验结果
    # expert_policy相当于是导入了一个专家到你的程序来, 以后你想实时问专家问题, 就用它的策略去生成答案
    parser.add_argument('--expert_policy_file', '-epf', type=str, required=True)  # relative to where you're running this script from
    
    # expert_data是专家数据, 在DAgger里可作为初始训练数据
    parser.add_argument('--expert_data', '-ed', type=str, required=True) #relative to where you're running this script from
    
    # env_name是环境的名字, 即在gym中创建的env的名字
    parser.add_argument('--env_name', '-env', type=str, help=f'choices: {", ".join(MJ_ENV_NAMES)}', required=True)
    
    # exp_name是此时实验的名字, 用于保存此时实验的日志和记录文件
    parser.add_argument('--exp_name', '-exp', type=str, default='pick an experiment name', required=True)

    # do_dagger是问你跑的这次实验是不是DAgger算法
    parser.add_argument('--do_dagger', action='store_true')

    # n_iter是训练轮数
    '''
    逻辑是这样的：
    进行 n_iter 次训练, 朴素BC只需要训练一轮, 用的数据集是 expert_data
    但是DAgger至少训练两轮, 第一轮用的数据集是 expert_data,
    从第二轮开始, 就要依靠目前的policy去采样。每次采样长度为ep_len, 直到采样长度>=batch_size时停止
    然后将采样轨迹的label用专家重新打好, 然后加入到replay_buffer里

    每轮训练都要进行 num_agent_train_steps_per_iter 次优化, 每次优化就从replay_buffer里任选 train_batch_size 个数据出来训练
    '''
    parser.add_argument('--n_iter', '-n', type=int, default=1)

    # ep_len就是episode length, 即每次采样的episode的最大长度
    parser.add_argument('--ep_len', type=int)

    # num_agent_train_steps_per_iter, 意思就是每轮训练要优化的次数
    parser.add_argument('--num_agent_train_steps_per_iter', type=int, default=1000)  # number of gradient steps for training policy (per iter in n_iter)

    # batch_size, 意思就是每次训练采样轨迹长度的下限
    parser.add_argument('--batch_size', type=int, default=1000)  # training data collected (in the env) during each iteration
    
    # train_batch_size意思是每次做梯度下降优化时, 从replay_buffer中采样的数据数
    parser.add_argument('--train_batch_size', type=int,
                        default=100)  # number of sampled data points to be used per gradient/train step
    
    # eval_batch_size, 意思是每次评估时要用的数据数量
    parser.add_argument('--eval_batch_size', type=int,
                        default=1000)  # eval data collected (in the env) for logging metrics

    # 神经网络中隐藏层的层数
    parser.add_argument('--n_layers', type=int, default=2)  # depth, of policy to be learned
    
    # 每层隐藏层的神经元个数
    parser.add_argument('--size', type=int, default=64)  # width of each layer, of policy to be learned
    
    # 学习率
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)  # LR for supervised learning

    # 保存视频的频率。一共要进行n_iter次训练, 这个参数意思就是进行多少次训练的时候保存一次视频
    parser.add_argument('--video_log_freq', type=int, default=5)

    # 保存指标的频率
    parser.add_argument('--scalar_log_freq', type=int, default=1)
    
    # 是否不使用gpu
    parser.add_argument('--no_gpu', '-ngpu', action='store_true')
    
    # gpu的号码, 默认使用第0个gpt
    parser.add_argument('--which_gpu', type=int, default=0)
    
    # replay_buffer的最大容量
    parser.add_argument('--max_replay_buffer_size', type=int, default=1000000)
    
    # 是否要保存每轮训练的参数
    parser.add_argument('--save_params', action='store_true')
    
    # 种子
    parser.add_argument('--seed', type=int, default=1)
    
    # 将命令行中的参数解析并存储在 args 中
    args = parser.parse_args()

    # convert args to dictionary
    # 将 args 转为字典, 保存为 params
    params = vars(args)

    ##################################
    ### CREATE DIRECTORY FOR LOGGING
    ##################################

    if args.do_dagger:
        # Use this prefix when submitting. The auto-grader uses this prefix.
        # 如果是DAgger算法，则日志目录使用q2_前缀。而且必须要保证至少两次训练，因为第一次是初始化训练，第二次是通过专家进行改进
        logdir_prefix = 'q2_'
        assert args.n_iter>1, ('DAGGER needs more than 1 iteration (n_iter>1) of training, to iteratively query the expert and train (after 1st warmstarting from behavior cloning).')
    else:
        # Use this prefix when submitting. The auto-grader uses this prefix.
        # 如果是朴素BC算法，则日志使用q1_前缀。而且训练一次即可，因为你到了专家数据然后模仿学习训练一次就好了
        logdir_prefix = 'q1_'
        assert args.n_iter==1, ('Vanilla behavior cloning collects expert data just once (n_iter=1)')

    # directory for logging
    # os.path.realpath(__file__) 获取当前 Python 脚本的绝对路径。
    # os.path.dirname() 用来获取当前文件的目录。
    # ../../data 是一个相对路径, 表示当前脚本所在目录的父目录的父目录中的 data 文件夹（即两个层级上级目录）。
    # 所以相当于data_path在hw1/data下, 即在hw1/下创建一个名为data的目录用于存储日志
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../data')
    if not (os.path.exists(data_path)):
        os.makedirs(data_path)
    
    # 本次实验日志文件夹名字 = 是否使用DAgger + 本次实验名字 + 本次实验采用的gym环境名字 + 时间戳
    logdir = logdir_prefix + args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    # 这里得到本次实验日志文件夹的绝对路径
    logdir = os.path.join(data_path, logdir)
    params['logdir'] = logdir
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    ###################
    ### RUN TRAINING
    ###################

    run_training_loop(params)


if __name__ == "__main__":
    main()
