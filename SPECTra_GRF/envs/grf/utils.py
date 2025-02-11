import numpy as np 


def _get_avail(obs, ball_distance):
    avail = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    (
        NO_OP,
        LEFT,
        TOP_LEFT,
        TOP,
        TOP_RIGHT,
        RIGHT,
        BOTTOM_RIGHT,
        BOTTOM,
        BOTTOM_LEFT,
        LONG_PASS,
        HIGH_PASS,
        SHORT_PASS,
        SHOT,
        SPRINT,
        RELEASE_MOVE,
        RELEASE_SPRINT,
        SLIDE,
        DRIBBLE,
        RELEASE_DRIBBLE,
    ) = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18)

    if obs["ball_owned_team"] == 1:  # opponents owning ball
        (
            avail[LONG_PASS],
            avail[HIGH_PASS],
            avail[SHORT_PASS],
            avail[SHOT],
            avail[DRIBBLE],
        ) = (0, 0, 0, 0, 0)
        if ball_distance > 0.03:
            avail[SLIDE] = 0
    elif (
        obs["ball_owned_team"] == -1
        and ball_distance > 0.03
        and obs["game_mode"] == 0
    ):  # Ground ball  and far from me
        (
            avail[LONG_PASS],
            avail[HIGH_PASS],
            avail[SHORT_PASS],
            avail[SHOT],
            avail[DRIBBLE],
            avail[SLIDE],
        ) = (0, 0, 0, 0, 0, 0)
    else:  # my team owning ball
        avail[SLIDE] = 0
        if ball_distance > 0.03:
            (
                avail[LONG_PASS],
                avail[HIGH_PASS],
                avail[SHORT_PASS],
                avail[SHOT],
                avail[DRIBBLE],
            ) = (0, 0, 0, 0, 0)

    # Dealing with sticky actions
    sticky_actions = obs["sticky_actions"]
    if sticky_actions[8] == 0:  # sprinting
        avail[RELEASE_SPRINT] = 0

    if sticky_actions[9] == 1:  # dribbling
        avail[SLIDE] = 0
    else:
        avail[RELEASE_DRIBBLE] = 0

    if np.sum(sticky_actions[:8]) == 0:
        avail[RELEASE_MOVE] = 0

    # if too far, no shot
    ball_x, ball_y, _ = obs["ball"]
    if ball_x < 0.64 or ball_y < -0.27 or 0.27 < ball_y:
        avail[SHOT] = 0
    elif (0.64 <= ball_x and ball_x <= 1.0) and (
        -0.27 <= ball_y and ball_y <= 0.27
    ):
        avail[HIGH_PASS], avail[LONG_PASS] = 0, 0

    if obs["game_mode"] == 2 and ball_x < -0.7:  # Our GoalKick
        avail = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        avail[LONG_PASS], avail[HIGH_PASS], avail[SHORT_PASS] = 1, 1, 1
        return avail

    elif obs["game_mode"] == 4 and ball_x > 0.9:  # Our CornerKick
        avail = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        avail[LONG_PASS], avail[HIGH_PASS], avail[SHORT_PASS] = 1, 1, 1
        return avail

    elif obs["game_mode"] == 6 and ball_x > 0.6:  # Our PenaltyKick
        avail = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        avail[SHOT] = 1
        return avail

    return avail




class Rewarder:
    def __init__(self, n_agents) -> None:
        self.n_agents = n_agents

    def calc_reward(self, r, prev_obs, obs):
        
        ball_position = obs[88:91] 
        self.ball_owned_team = np.where(obs[94:97] == 1)[0][0]

        add_reward = (
            ball_position_reward(ball_position)
            + self.posession_reward()
        )

        return add_reward + r

    def posession_reward(self):
        if self.ball_owned_team == 1:
            return 0.001
        
        elif self.ball_owned_team == 2:
            return -0.001
        else:
            return 0.0

def ball_position_reward(ball_position):
    ball_x, ball_y, _ = ball_position
    MIDDLE_X, PENALTY_X, END_X = 0.2, 0.64, 1.0
    PENALTY_Y, END_Y = 0.27, 0.42
    ball_position_r = 0.0
    if (-END_X <= ball_x and ball_x < -PENALTY_X) and (
        -PENALTY_Y < ball_y and ball_y < PENALTY_Y
    ):
        ball_position_r = -0.0015
    elif (-END_X <= ball_x and ball_x < -MIDDLE_X) and (
        -END_Y < ball_y and ball_y < END_Y
    ):
        ball_position_r = -0.001
    elif (-MIDDLE_X <= ball_x and ball_x <= MIDDLE_X) and (
        -END_Y < ball_y and ball_y < END_Y
    ):
        ball_position_r = -0.0005
    elif (PENALTY_X < ball_x and ball_x <= END_X) and (
        -PENALTY_Y < ball_y and ball_y < PENALTY_Y
    ):
        ball_position_r = 0.0005
    elif (MIDDLE_X < ball_x and ball_x <= END_X) and (
        -END_Y < ball_y and ball_y < END_Y
    ):
        ball_position_r = 0.0
    else:
        ball_position_r = -0.0005

    return ball_position_r