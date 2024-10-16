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

%% fit normal dist

figure
for i = 1:10
    subplot(2,5,i)
    x = linspace(min(VAR_DATA(:,i)),max(VAR_DATA(:,i)),1000);
    histogram(VAR_DATA(var_diagnosis,i),'Normalization','pdf','FaceColor',[.2,.2,.8]);hold on
    histogram(VAR_DATA(var_diagnosis,i),'Normalization','pdf','FaceColor',[.8,.2,.2]);
    [mu,sigma] = normfit(VAR_DATA(var_diagnosis,i));
    y_M = normpdf(x,mu,sigma);
    [mu,sigma] = normfit(VAR_DATA(~var_diagnosis,i));
    y_B = normpdf(x,mu,sigma);
    plot(x,y_B,'Color','blue','LineWidth',2);hold on
    plot(x,y_M,'Color','red','LineWidth',2);
    if i>5
    xlabel('Value');
    end
    if mod(i,5) == 1
    ylabel('PDF');
    end
    title(erase(strrep(vnames{i},'_',' '),'1'))
end

%% Histogram
figure
for i = 1:10
    subplot(2,5,i)
    x = linspace(min(VAR_DATA(:,i)),max(VAR_DATA(:,i)),1000);
    histogram(VAR_DATA(var_diagnosis,i),'Normalization','pdf','FaceColor',[.2,.2,.8]);hold on
    histogram(VAR_DATA(~var_diagnosis,i),'Normalization','pdf','FaceColor',[.8,.2,.2]);
    title(erase(strrep(vnames{i},'_',' '),'1'))
end

figure;
mean_b = nan(1,10);
mean_m = nan(1,10);
for i = 1:10
    subplot(2,5,i);

    
    data = rescale(VAR_DATA(:,i),1,10,"InputMax",max(VAR_DATA(:,i)),"InputMin",min(VAR_DATA(:,i)));

    x = linspace(min(data),max(data),1000);

    [N,edges] = histcounts(data(var_diagnosis));
    N = N/max(N);
    histogram('BinCounts',N,'BinEdges',edges,'FaceColor','red','FaceAlpha',0.5,'Normalization','pdf');hold on;
    [N,edges] = histcounts(data(~var_diagnosis));
    N = N/max(N);
    histogram('BinCounts',N,'BinEdges',edges,'FaceColor','blue','FaceAlpha',0.3,'Normalization','pdf');hold on;

    [mu,sigma] = normfit(data(var_diagnosis));
    y_M = normpdf(x,mu,sigma);
    [mu,sigma] = normfit(data(~var_diagnosis));
    y_B = normpdf(x,mu,sigma);
%     plot(x,y_M/max(y_M),'Color','red','LineWidth',2);hold on
%     plot(x,y_B/max(y_B),'Color','blue','LineWidth',2);hold on
    plot(x,y_M,'Color','red','LineWidth',2);hold on
    plot(x,y_B,'Color','blue','LineWidth',2);hold on


%     histogram('BinCounts',N,'BinEdges',edges,'FaceColor','blue');
    title(erase(strrep(VariableNames{i+2},'_',' '),'1'));
% 
%     if i>5
%     xlabel('Value');
%     end
    if mod(i,5) == 1
    ylabel('Probability');
    end
    xlim([0,10])
    
end