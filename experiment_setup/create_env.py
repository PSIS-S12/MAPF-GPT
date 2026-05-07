from gymnasium import Wrapper
from loguru import logger
from pogema import AnimationConfig, AnimationMonitor, pogema_v0
from pogema.wrappers.metrics import RuntimeMetricWrapper
from pogema_toolbox.create_env import MultiMapWrapper


class LogActions(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.made_actions = None
        self.init_positions = None

    def step(self, actions):
        observations, rewards, terminated, truncated, infos = self.env.step(actions)
        for i, action in enumerate(actions):
            self.made_actions[i].append(action)

        if all(terminated) or all(truncated):
            infos[0]["metrics"]["made_actions"] = self.made_actions
            infos[0]["metrics"]["init_positions"] = self.init_positions

        return observations, rewards, terminated, truncated, infos

    def reset(self, **kwargs):
        observations, info = self.env.reset(**kwargs)
        self.made_actions = [[] for _ in observations]
        self.init_positions = [obs["global_xy"] for obs in observations]

        if self.unwrapped.grid_config.on_target == "restart":
            self.global_lifelong_targets_xy = [
                [[int(x), int(y)] for x, y in obs["global_lifelong_targets_xy"]]
                for obs in observations
            ]

        return observations, info


class CollisionCounterWrapper(Wrapper):
    """
    Steje poskuse konfliktov med agenti PREDEN jih collision_system='soft' popravi.

    Meri:
    - vertex collisions: dva ali vec agentov zeli v isto celico,
    - edge collisions: dva agenta zamenjata mesti v istem koraku.

    To je bolj uporabno kot gledanje koncnih pozicij po env.step(), ker soft collision
    system konflikte popravi tako, da agenti ostanejo na mestu.
    """

    # Standardna POGEMA akcijska shema:
    # 0 = wait, 1 = up, 2 = down, 3 = left, 4 = right
    DEFAULT_MOVES = {
        0: (0, 0),
        1: (-1, 0),
        2: (1, 0),
        3: (0, -1),
        4: (0, 1),
    }

    def __init__(self, env):
        super().__init__(env)
        self.collisions = 0
        self.prev_positions = None

    def reset(self, **kwargs):
        observations, info = self.env.reset(**kwargs)
        self.collisions = 0
        self.prev_positions = self._positions_from_observations(observations)
        return observations, info

    def step(self, actions):
        if self.prev_positions is not None:
            intended_positions = self._get_intended_positions(self.prev_positions, actions)

            vertex_collisions = self._count_vertex_collisions(intended_positions)
            edge_collisions = self._count_edge_collisions(self.prev_positions, intended_positions)

            self.collisions += vertex_collisions + edge_collisions

        observations, rewards, terminated, truncated, infos = self.env.step(actions)

        self.prev_positions = self._positions_from_observations(observations)

        if all(terminated) or all(truncated):
            self._write_metric(infos)

        return observations, rewards, terminated, truncated, infos

    def _positions_from_observations(self, observations):
        return [tuple(obs["global_xy"]) for obs in observations]

    def _get_intended_positions(self, positions, actions):
        intended = []

        for (x, y), action in zip(positions, actions):
            dx, dy = self.DEFAULT_MOVES[int(action)]
            intended.append((x + dx, y + dy))

        return intended

    @staticmethod
    def _count_vertex_collisions(positions):
        return len(positions) - len(set(positions))

    @staticmethod
    def _count_edge_collisions(prev_positions, intended_positions):
        count = 0
        n = len(prev_positions)

        for i in range(n):
            for j in range(i + 1, n):
                if prev_positions[i] == intended_positions[j] and prev_positions[j] == intended_positions[i]:
                    count += 1

        return count

    def _write_metric(self, infos):
        if isinstance(infos, list) and len(infos) > 0:
            infos[0].setdefault("metrics", {})
            infos[0]["metrics"]["collisions"] = self.collisions
        elif isinstance(infos, dict):
            infos.setdefault("metrics", {})
            infos["metrics"]["collisions"] = self.collisions


def create_eval_env(config):
    env = pogema_v0(grid_config=config)

    # Omogoca uporabo vec map iz maps.yaml in grid_search map_name.
    env = MultiMapWrapper(env)

    # Dodana metrika: collisions.
    # Mora biti pred RuntimeMetricWrapper, da runtime wrapper potem doda se cas izvajanja.
    env = CollisionCounterWrapper(env)

    # Standardne metrike: ISR, CSR, ep_length, SoC, makespan, runtime.
    env = RuntimeMetricWrapper(env)

    if config.with_animation:
        logger.debug("Wrapping environment with AnimationMonitor")
        env = AnimationMonitor(env, AnimationConfig(save_every_idx_episode=None))

    return env


def create_logging_env(config):
    env = pogema_v0(grid_config=config)

    env = MultiMapWrapper(env)
    env = LogActions(env)
    env = CollisionCounterWrapper(env)

    if config.with_animation:
        logger.debug("Wrapping environment with AnimationMonitor")
        env = AnimationMonitor(env, AnimationConfig(save_every_idx_episode=None))

    env = RuntimeMetricWrapper(env)

    return env
