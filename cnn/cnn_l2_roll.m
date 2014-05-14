function theta = cnn_l2_roll(weights)

theta = [];
theta = [theta ; weights.vishid(:)];
theta = [theta ; weights.hidbias_bu(:)];
theta = [theta ; weights.hidpen(:)];
theta = [theta ; weights.penbias_bu(:)];

theta = [theta ; weights.penhyper(:)];
theta = [theta ; weights.hyperbias(:)];
theta = [theta ; weights.hyperpen(:)];
theta = [theta ; weights.penbias_td(:)];

theta = [theta ; weights.penhid(:)];
theta = [theta ; weights.hidbias_td(:)];
theta = [theta ; weights.hidvis(:)];
theta = [theta ; weights.visbias(:)];

return;