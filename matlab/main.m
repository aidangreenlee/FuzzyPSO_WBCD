data = readtable('data/diagnostic.data','FileType','text');
VariableNames = {'ID','Diagnosis','radius1','texture1','perimeter1','area1','smoothness1','compactness1','concavity1','concave_points1',...
    'symmetry1','fractal_dimension1','radius2','texture2','perimeter2','area2','smoothness2','compactness2','concavity2','concave_points2',...
    'symmetry2','fractal_dimension2','radius3','texture3','perimeter3','area3','smoothness3','compactness3','concavity3','concave_points3',...
    'symmetry3','fractal_dimension3'};
data.Properties.VariableNames = VariableNames;

i_val = contains(VariableNames,{'1'});%,'2','3'})% | contains(VariableNames,'Diagnosis')% & ~contains(VariableNames,'radius') & ~contains(VariableNames,'perimeter')% & ~contains(VariableNames,'area');
vnames = VariableNames(i_val);
e_val = contains(VariableNames,{'2'});
VAR_DATA = table2array(data(:,i_val));
e_data = table2array(data(:,e_val));
var_diagnosis = contains(table2array(data(:,2)),'M');
%%

VAR_DATA = VAR_DATA./max(VAR_DATA)*10;
ntrain = 379;
rng(1337)
rand_idx = randperm(length(VAR_DATA));
training_data = VAR_DATA(rand_idx(1:ntrain),1:end);
training_out  = var_diagnosis(rand_idx(1:ntrain));
testing_data = VAR_DATA(rand_idx(ntrain+1:end),1:end);
testing_out  = var_diagnosis(rand_idx(ntrain+1:end));


x = training_data; % input data
% x = VAR_DATA./max(VAR_DATA)*10; % rescale data between 0 and 10;


%%
c=5;
KMeans
% FuzzyMFs;
%%


% Now do particle swarm
P = PSO(r, 20);
P.parameter_Vmax = 3;
% P.parameter_Vmax = 2;s
P.parameter_alpha = 0.2;
P.parameter_Beta = 0.4;
P.parameter_gamma = 0.4;
P.parameter_W = 1;
P.parameter_phi1 = 0;
P.parameter_phi2 = 1;
y_hat = nan(length(training_data),1);
c = 0; % count number of loops
H = [];
Hmin = [];
Hmax = [];
Hmean = [];
Hmedian = [];
Hq1 = [];
Hq3 = [];
tmp1 = [];
tmp2 = [];
% while P.particle_best.H(end) < .98

clear Particle_Positions
clear Particle_Velocities
[Particle_Positions(:,:,:,1), Particle_Velocities(:,:,:,1)] = getParticlePositions(P);

W = linspace(3,0.1,400);
Phi1 = linspace(4,1,400);
Phi2 = linspace(2,5,400);

while c <= 201 || length(unique(H(end-200:end))) ~= 1
%     if c >= 500 || any([P.particle.Hbest] > .98)
%         break
%     end
%     if (c+1) >= 400
%         P.parameter_W = .1;
% %         P.parameter_phi1 = 1;
% %         P.parameter_phi2 = 4;
%     else
%         P.parameter_W = W(c+1);
% %         P.parameter_phi1 = Phi1(c+1);
% %         P.parameter_phi2 = Phi2(c+1);
%     end
    for m = 1:P.parameter_M
        for q = 1:length(training_data)
            y_hat(q) = P.calculate_NFS(P.particle(m).P,training_data(q,:)');
        end

        P.particle(m).H = P.H(y_hat,training_out);
        P = P.update(m);
%         H(m)=P.particle(m).H(end);
        



    end
%     P.particle_best.H(end+1) = max(H);
%     idx = find(H == P.particle_best.H(end),1, 'last');
%     P.particle_best.Pbest_g = P.particle(idx).P;
    disp(P.particle_best.H(end))
    c = c + 1;
    %% save particle data
    [Particle_Positions(:,:,:,c), Particle_Velocities(:,:,:,c)] = getParticlePositions(P);

    %%
    H(end+1)  = P.particle_best.H(end);
%     tmp1(end+1,:) = P.particle_best.Pbest_g(7:end,1);
%     tmp2(end+1,:) = P.particle_best.Pbest_g(7:end,2);

    temp = [P.particle.H];
    itemp = ~isinf(temp);
    Hmax(end+1) = max(temp(itemp));
    Hmin(end+1) = min(temp(itemp));
    Hmean(end+1) = mean(temp(itemp));
    Hmedian(end+1) = median(temp(itemp));
end

%% Now run testing data
y_hat = [];
for q = 1:length(testing_data)
    y_hat(q) = P.calculate_NFS(P.particle_best.Pbest_g,testing_data(q,:)');
end
[~,accuracy] = P.H(y_hat,testing_out);
fprintf('Accuracy = %.02f%% in %d loops\n',accuracy*100,c)

Cs = P.particle_best.Pbest_g(1:P.n,:);
sigmas = P.particle_best.Pbest_g(P.n+1:2*P.n,:);
omegas = P.particle_best.Pbest_g(2*P.n+1:end,:);
x_range = 0:.05:10;
figure;
for j = 1:P.J
    p(j) = subplot(P.J,1,j);
    for i = 1:size(Cs,1)
%         plot(x,normpdf(x,Cs(i,j),sigmas(i,j)),'LineWidth',2);hold on;
        plot(x_range,exp(-(x_range-Cs(i,j)).^2/sigmas(i,j).^2),'LineWidth',2);hold on;
    end
    grid on;
    ylabel('Membership')
end
subplot(P.J,1,1);
legend(VariableNames(2:end),'Location','best');
% linkaxes(p)


figure;
plot(H,'LineWidth',2);grid on;
title('Best Particle Fitness')
xlabel('Iterations');
ylabel('H')

figure;
for m = 1:P.parameter_M
    plot(P.particle(m).H,'LineWidth',1);hold on
end
plot(H,'-','Color','Red','LineWidth',3);grid on;hold on;
xlabel('Iteration');ylabel('H')

figure;
plot(H,'-','Color','Red','LineWidth',3);grid on;hold on;
plot(Hmin,'Color','Cyan','LineWidth',3);
plot(Hmean,'Color','Green','LineWidth',3);
plot(Hmedian,'--','Color',[0,.8,0],'LineWidth',3)
plot(Hmax,'--','Color','Blue','LineWidth',3);
legend({'Particle Best', 'Minimum Fitness','Average Fitness','Median Fitness','Maximum Fitness'},'Location','best')
ylim([0,1]);
xlabel('Iteration');
ylabel('H')


figure;
% animateParticles(Particle_Positions,1,7, {'radius','concavity'})
plotParticle(Particle_Positions,5)
xlabel('Iteration');
ylabel('MF center')
set(gcf, 'Position',[680 533 869 445]);
set(gca, 'FontSize',16);grid on
legend(strrep(erase(vnames,'1'),'_',' '),'Location','bestoutside','FontSize',12)
title('Particle 1')


figure;
plotParticle(Particle_Velocities,5)
xlabel('Iteration');
ylabel('MF center')
set(gcf, 'Position',[680 533 869 445]);
set(gca, 'FontSize',16);grid on
legend(strrep(erase(vnames,'1'),'_',' '),'Location','bestoutside','FontSize',12)
title('Particle 1 Velocity')