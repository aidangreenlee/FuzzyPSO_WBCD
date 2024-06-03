data = importdata("../julia/output1.txt",",",2);
output = data.data;
pSize = erase(data.textdata{1}, 'particleSize: ');

parameters = split(data.textdata{2}, ', ');
W = str2double(erase(parameters{1},'W:'));
phi_1 = str2double(erase(parameters{2},'phi_1:'));
phi_2 = str2double(erase(parameters{3},'phi_2:'));

pSize = str2num(pSize);
posSize = (pSize - 1)/2;

overall_fitness = importdata("../julia/fitness.txt");

%%
nCluster = 1;
index = 1;
fitness = 2;
% position = 3:(fitness + nCluster*pSize);
position = 3:(fitness+nCluster*pSize*3);
velocity = (position(end) + 1):size(output,2);


%%

figure;
subplot(1,3,1)
plot(output(:,position(1:(pSize-1)/2)));
% plot(output(:,position));
ylim([-50,50])
xlabel('Iteration');
ylabel('Position');
title('Particle Position');

subplot(1,3,2)
plot(output(:,velocity(1:(pSize-1)/2)));
% plot(output(:,velocity)); 
ylim([-50,50])
xlabel('Iteration');
ylabel('Velocity');
title('Particle Velocity');

subplot(1,3,3)
% plot(output(:, fitness));
plot(overall_fitness(:,1),'DisplayName','Minimum fitness'); hold on
plot(overall_fitness(:,2),'DisplayName','Maximum fitness');
plot(overall_fitness(:,3),'DisplayName','Average fitness');
plot(overall_fitness(:,4),'DisplayName','Median fitness');
plot(overall_fitness(:,5),'DisplayName','Global best fitness');
ylabel('Fitness');
xlabel('Iteration')
ylim([0,1])
title('Particle Fitness');
legend
sgtitle(sprintf('W:%0.3f \\phi_1:%0.3f \\phi_2:%0.3f',W, phi_1, phi_2))

set(gcf,'Position',[372 377 1191 420])
