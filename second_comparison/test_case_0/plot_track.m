forces = dlmread('for.track');
positions = dlmread('pos.track');
velocities = dlmread('vel.track');
figure(1)
plot(forces)
title('forces')
figure(2)
plot(positions)
title('positions')
figure(3)
plot(velocities)
title('velocities')
