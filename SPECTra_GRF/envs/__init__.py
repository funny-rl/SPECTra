from functools import partial

from .multiagentenv import MultiAgentEnv
from .grf import Academy_3_vs_1_with_Keeper, Academy_Pass_and_Shoot_with_Keeper, Academy_Run_Pass_and_Shoot_with_Keeper

import sys
import os


def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)


REGISTRY = {
    # "sc2": partial(env_fn, env=StarCraft2Env),
    # "matrix_game_1": partial(env_fn, env=Matrix_game1Env),
    # "matrix_game_2": partial(env_fn, env=Matrix_game2Env),
    # "matrix_game_3": partial(env_fn, env=Matrix_game3Env),
    # "mmdp_game_1": partial(env_fn, env=mmdp_game1Env)
    "academy_3_vs_1_with_keeper": partial(env_fn, env=Academy_3_vs_1_with_Keeper),
    "academy_pass_and_shoot_with_keeper": partial(env_fn, env=Academy_Pass_and_Shoot_with_Keeper),
    "academy_run_pass_and_shoot_with_keeper": partial(env_fn, env=Academy_Run_Pass_and_Shoot_with_Keeper),
    
}


if sys.platform == "linux":
    os.environ.setdefault("SC2PATH",
                          os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))
