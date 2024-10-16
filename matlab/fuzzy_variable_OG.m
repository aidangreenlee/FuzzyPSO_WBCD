data = readtable('data/original.data','FileType','text');
VariableNames = {'ID','Clump_Thickness','Uniformity_of_Cell_Size','Uniformity_of_Cell_Shape','Marginal_Adhesion',...
    'Single_Epithelial_Cell_Size','Bare_Nuclei','Bland_Chromatin','Normal_Nucleoli','Mitoses',...
    'Diagnosis'};
data.Properties.VariableNames = VariableNames;

% i_val = contains(VariableNames,{'1'});%,'2','3'})% | contains(VariableNames,'Diagnosis')% & ~contains(VariableNames,'radius') & ~contains(VariableNames,'perimeter')% & ~contains(VariableNames,'area');
vnames = VariableNames(2:end);
e_val = contains(VariableNames,{'2'});
VAR_DATA = table2array(data(:,2:end));
% e_data = table2array(data(:,e_val));
% var_diagnosis = contains(table2array(data(:,2)),'4');
var_diagnosis = data.Diagnosis == 4;

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

    
%     data = rescale(VAR_DATA(:,i),1,10,"InputMax",max(VAR_DATA(:,i)),"InputMin",min(VAR_DATA(:,i)));
    data = VAR_DATA(:,i);
    data(isnan(data)) = [];
    vdiag = var_diagnosis(~isnan(VAR_DATA(:,i)));
    x = linspace(min(data),max(data),1000);

    [N,edges] = histcounts(data(vdiag));
    N = N/max(N);
    histogram('BinCounts',N,'BinEdges',edges,'FaceColor','red','FaceAlpha',0.5,'Normalization','pdf');hold on;
    [N,edges] = histcounts(data(~vdiag));
    N = N/max(N);
    histogram('BinCounts',N,'BinEdges',edges,'FaceColor','blue','FaceAlpha',0.3,'Normalization','pdf');hold on;

    [mu,sigma] = normfit(data(vdiag));
    y_M = normpdf(x,mu,sigma);
    [mu,sigma] = normfit(data(~vdiag));
    y_B = normpdf(x,mu,sigma);
%     plot(x,y_M/max(y_M),'Color','red','LineWidth',2);hold on
%     plot(x,y_B/max(y_B),'Color','blue','LineWidth',2);hold on
    plot(x,y_M,'Color','red','LineWidth',2);hold on
    plot(x,y_B,'Color','blue','LineWidth',2);hold on


%     histogram('BinCounts',N,'BinEdges',edges,'FaceColor','blue');
    title(erase(strrep(VariableNames{i+1},'_',' '),'1'));
% 
%     if i>5
%     xlabel('Value');
%     end
    if mod(i,5) == 1
    ylabel('Probability');
    end
    xlim([0,10])
    
end