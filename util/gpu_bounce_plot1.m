

src_dir = '../ppmd/output/';


p0 = dlmread(strcat(src_dir,'pos0.track'));
v0 = dlmread(strcat(src_dir,'vel0.track'));
f0 = dlmread(strcat(src_dir,'for0.track'));

p1 = dlmread(strcat(src_dir,'pos1.track'));
v1 = dlmread(strcat(src_dir,'vel1.track'));
f1 = dlmread(strcat(src_dir,'for1.track'));

figure(1)
subplot(1,2,1)

title('positions 0')
plot(p0)

subplot(1,2,2)

title('positions 1')
plot(p1)


figure(2)
subplot(1,2,1)

title('velocities 0')
plot(v0)

subplot(1,2,2)

title('velocities 1')
plot(v1)


figure(3)
subplot(1,2,1)

title('forces 0')
plot(f0)

subplot(1,2,2)

title('forces 1')
plot(f1)


