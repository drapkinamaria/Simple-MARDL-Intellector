import intellector.pieces as pieces

WIN_REWARD = 1000
LOSE_REWARD = -1000
DRAW_REWARD = -100
DRAW = 0
PROMOTION_REWARD = 200
CAPTURE_REWARDS = {
    pieces.DEFENSOR: 100,
    pieces.AGRESSOR: 100,
    pieces.DOMINATOR: 100,
    pieces.LIBERATOR: 100,
    pieces.PROGRESSOR: 50,
    pieces.INTELLECTOR: 300,
}
