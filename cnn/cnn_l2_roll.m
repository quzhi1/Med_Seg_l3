function theta = cnn_l2_roll(weights)

theta = [];
theta = [theta ; weights.vishid(:)];
theta = [theta ; weights.hidbias_bu(:)];
theta = [theta ; weights.hidpen(:)];
theta = [theta ; weights.penbias(:)];
theta = [theta ; weights.penhid(:)];
theta = [theta ; weights.hidbias_td(:)];
theta = [theta ; weights.hidvis(:)];
theta = [theta ; weights.visbias(:)];

return;