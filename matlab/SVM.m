% data = readtable('data/diagnostic.data','FileType','text');
% VariableNames = {'ID','Diagnosis','radius1','texture1','perimeter1','area1','smoothness1','compactness1','concavity1','concave_points1',...
%              'symmetry1','fractal_dimension1','radius2','texture2','perimeter2','area2','smoothness2','compactness2','concavity2','concave_points2',...
%              'symmetry2','fractal_dimension2','radius3','texture3','perimeter3','area3','smoothness3','compactness3','concavity3','concave_points3',...
%              'symmetry3','fractal_dimension3'};
% data.Properties.VariableNames = VariableNames;
% 
% i_val = contains(VariableNames,{'1','2','3'});% | contains(VariableNames,'Diagnosis')% & ~contains(VariableNames,'radius') & ~contains(VariableNames,'perimeter')% & ~contains(VariableNames,'area');
% 
% var_data = table2array(data(:,i_val));
% var_diagnosis = double(contains(table2array(data(:,2)),'M'));
% var_diagnosis(~var_diagnosis) = -1;
% % 
% mu = mean(var_data)';
% n = size(var_data,1);

%%
data = readtable('./data/diagnostic.data','FileType','text');
VariableNames = {'ID','Diagnosis','radius1','texture1','perimeter1','area1','smoothness1','compactness1','concavity1','concave_points1',...
             'symmetry1','fractal_dimension1','radius2','texture2','perimeter2','area2','smoothness2','compactness2','concavity2','concave_points2',...
             'symmetry2','fractal_dimension2','radius3','texture3','perimeter3','area3','smoothness3','compactness3','concavity3','concave_points3',...
             'symmetry3','fractal_dimension3'};
data.Properties.VariableNames = VariableNames;
results = contains(data.Diagnosis,'M');
data = table2array(data(:,[3:end]));
idx = contains(VariableNames,'1');
data = data(:,idx);
data = [data,results];

ntrain = 379;
rng(1337)

% ntrain = 455;
data(logical(sum(isnan(data),2)),:) = [];
data(data(:,end)==2,end)=0;
data(data(:,end)==4,end)=1;
data(:,1:end-1) = (data(:,1:end-1)-min(data(:,1:end-1)))./range(data(:,1:end-1))*10;
rand_idx = randperm(length(data));
training_data = data(rand_idx(1:ntrain),2:end-1);
training_out  = data(rand_idx(1:ntrain),end);
testing_data = data(rand_idx(ntrain+1:end),2:end-1);
testing_out  = data(rand_idx(ntrain+1:end),end);
%%

mu = mean(training_data)';
n = size(training_data,1);
% Normalize B
V = training_data - ones(n,1) * mu';

for i = 1:size(V,2)
    V(:,i) = (V(:,i))/std(V(:,i));
end

%% SVM
RBF = @(x1, x2, gamma) exp(-gamma * vecnorm(x1-x2).^2);
% 
% RBF(var_data, var_diagnosis, 1.3)

gamma = 1/(2*5);
K = [];
[A,B] = ndgrid(1:length(V),1:length(V));
for i = 1:size(A,1)
    for j = 1:size(A,2)
        K(i,j) = RBF(V(A(i,j),:),V(B(i,j),:),gamma);
    end
end
% figure;cdfplot(K)

%% Kernel PCA
oneN = ones(size(K))/size(K,1);
% calculate Gram matrix (make sure data has zero mean)
G = K - oneN*K - K*oneN + oneN*K*oneN;

% equation 27

[eVec,D] = eig(G);

d = diag(D);

[~,idx] = sort(d,"descend");
V_s = eVec(:,idx);

g = cumsum(d);

figure;
plot(0:length(g),[0;g./g(end)],'LineWidth',2)
% xticklabels([' ',erase(strrep(vnames(idx),'_',' '),'1')])
set(gca,'FontSize',12)
grid on;
ylabel('Cumulative Energy')
title('PCA Energy')
xtickangle(60)
ylim([0,1])
yline(0.95, 'Label','95 %');
%B = var_data - ones(n,1) * mu';

%%
a1 = V_s(:,1);
a2 = V_s(:,2);
xs = real(sum(a1.*G));
ys = real(sum(a2.*G));

figure;
scatter(xs(training_out==0), ys(training_out==0),[],'red','filled')
hold on
scatter(xs(training_out==1), ys(training_out==1),[],'blue','filled')
grid on
xlabel('P_1')
ylabel('P_2')
%% SVM
Beta = zeros(size(K));
y = training_out;
alpha = .01;
for i = 1:1000
    Beta = Beta + alpha*(y - K * Beta);
end

y_out = nan(size(y));
for j = 1:length(y)
    y_out(j) = round(sum(Beta(:,j).*K(:,j)));
end
sum(y_out ~= y);