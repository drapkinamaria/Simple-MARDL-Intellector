import intellector.pieces as pieces

WIN_REWARD = 1
LOSE_REWARD = -WIN_REWARD
DRAW_REWARD = -0.5
DRAW = 0
PROMOTION_REWARD = 0.2
CAPTURE_REWARDS = {
    pieces.DEFENSOR: 0.1,
    pieces.AGRESSOR: 0.2,
    pieces.DOMINATOR: 0.2,
    pieces.LIBERATOR: 0.1,
    pieces.PROGRESSOR: 0.05,
    pieces.INTELLECTOR: 0.35,
}
