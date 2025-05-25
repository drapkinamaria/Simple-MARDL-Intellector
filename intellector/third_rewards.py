import intellector.pieces as pieces

WIN_REWARD = 10
LOSE_REWARD = -WIN_REWARD
DRAW_REWARD = -5
DRAW = 0
CAPTURE_REWARDS = {
    pieces.INTELLECTOR: 3.5,
}
STEP_PENALTY = -0.1
