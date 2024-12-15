from cs285.infrastructure.utils import *


class ReplayBuffer(object):

    def __init__(self, max_size=1000000):

        self.max_size = max_size

        # store each rollout (这里的rollout就是指轨迹)
        self.paths = []

        # store (concatenated) component arrays from each rollout
        self.obs = None
        self.acs = None
        self.rews = None
        self.next_obs = None
        self.terminals = None

        # paths相当于若干条轨迹的集合, path是一个字典, 格式参见utils.py里的sample_trajectory
        # 然后obs, acs, rews, next_obs, terminals相当于拆出来的集合
        # 反正你想要完整的轨迹就调用self.paths, 想要分开的某个属性就调用self.xxs

    def __len__(self):
        if self.obs is not None:    # 这里我加了一个 is not None, 不然会报错
            return self.obs.shape[0]
        else:
            return 0

    # 这里传入的paths是若干条轨迹, concat_rew=True是拼接, 即将若干条轨迹的数据都拼接到一起
    def add_rollouts(self, paths, concat_rew=True):
        # add new rollouts into our list of rollouts
        for path in paths:
            self.paths.append(path)

        # convert new rollouts into their component arrays, and append them onto
        # our arrays
        observations, actions, rewards, next_observations, terminals = (
            convert_listofrollouts(paths, concat_rew)
        )

        if self.obs is None:    # 如果self.obs是空的，说明是第一次添加rollouts
            self.obs = observations[-self.max_size:]
            self.acs = actions[-self.max_size:]
            self.rews = rewards[-self.max_size:]
            self.next_obs = next_observations[-self.max_size:]
            self.terminals = terminals[-self.max_size:]
        else:
            self.obs = np.concatenate([self.obs, observations])[-self.max_size:]
            self.acs = np.concatenate([self.acs, actions])[-self.max_size:]
            if concat_rew:
                self.rews = np.concatenate(
                    [self.rews, rewards]
                )[-self.max_size:]
            else:
                if isinstance(rewards, list):
                    self.rews += rewards
                else:
                    self.rews.append(rewards)
                self.rews = self.rews[-self.max_size:]
            self.next_obs = np.concatenate(
                [self.next_obs, next_observations]
            )[-self.max_size:]
            self.terminals = np.concatenate(
                [self.terminals, terminals]
            )[-self.max_size:]

