import intellector.pieces as pieces

WIN_REWARD = 10
LOSE_REWARD = -WIN_REWARD
DRAW_REWARD = -5
DRAW = 0
PROMOTION_REWARD = 2
CAPTURE_REWARDS = {
    pieces.DEFENSOR: 1,
    pieces.AGRESSOR: 2,
    pieces.DOMINATOR: 2,
    pieces.LIBERATOR: 1,
    pieces.PROGRESSOR: 0.5,
    pieces.INTELLECTOR: 3.5,
}
