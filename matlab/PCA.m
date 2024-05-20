data = readtable('data/diagnostic.data','FileType','text');
VariableNames = {'ID','Diagnosis','radius1','texture1','perimeter1','area1','smoothness1','compactness1','concavity1','concave_points1',...
    'symmetry1','fractal_dimension1','radius2','texture2','perimeter2','area2','smoothness2','compactness2','concavity2','concave_points2',...
    'symmetry2','fractal_dimension2','radius3','texture3','perimeter3','area3','smoothness3','compactness3','concavity3','concave_points3',...
    'symmetry3','fractal_dimension3'};
data.Properties.VariableNames = VariableNames;

i_val = contains(VariableNames,{'1'});%,'2','3'})% | contains(VariableNames,'Diagnosis')% & ~contains(VariableNames,'radius') & ~contains(VariableNames,'perimeter')% & ~contains(VariableNames,'area');
e_val = contains(VariableNames,{'2'});
VAR_DATA = table2array(data(:,i_val));
e_data = table2array(data(:,e_val));
var_diagnosis = contains(table2array(data(:,2)),'M');
%%
vnames = VariableNames(i_val);

for itest = 1:4
    if itest == 1
        var_data = VAR_DATA;
        titlename = 'Mean';
    elseif itest == 2
        var_data = VAR_DATA - 3*e_data;
        titlename = '-3\sigma';
    elseif itest == 3
        var_data = VAR_DATA + 3*e_data;
        titlename = '+3\sigma';
    elseif itest == 4
        var_data = table2array(data(:,contains(VariableNames,{'3'})));
        titlename = 'Maximum';
    end

    mu = mean(var_data)';
    n = size(var_data,1);

    %% Normalize B
    B = var_data - ones(n,1) * mu';
    B_e = e_data - ones(n,1) * mean(e_data);

    for i = 1:size(B,2)
        B(:,i) = (B(:,i))/std(B(:,i));
    end
% 
%     figure;
%     boxplot(B)
%     xticklabels(erase(strrep(vnames,'_',' '),'1'))
%     set(gca,'FontSize',12)
%     grid on;
%     xtickangle(60)

    C = 1/(n-1) * ctranspose(B) * B;
    [V,D] = eig(C);

    d = diag(D);

    [~,idx] = sort(d,"descend");
    V_s = V(:,idx);

    g = cumsum(d);

    figure(199);
    plot(0:length(g),[0;g./g(end)],'LineWidth',2,'DisplayName',titlename); hold on
    xticklabels([' ',erase(strrep(vnames(idx),'_',' '),'1')])
    set(gca,'FontSize',12)
    grid on;
    ylabel('Cumulative Energy')
    title('PCA Energy')
    xtickangle(60)

end
L=legend;
L.AutoUpdate = 'off';
    ylim([0,1])
    yline(0.95, 'Label','95 %');
    
% %% Plot data
for ii = idx'
    figure;
    sgtitle(erase(strrep(vnames(ii),'_',' '),'1'))
    iloop = idx(idx~=ii)';
    for i = 1:length(iloop)
        subplot(3,3,i)
        scatter(B(var_diagnosis,ii),B(var_diagnosis,iloop(i)),[],'red')
        hold on
        scatter(B(~var_diagnosis,ii),B(~var_diagnosis,iloop(i)),[],'blue')
        ylabel(erase(strrep(vnames(iloop(i)),'_',' '),'1'))
    end
end

figure;
H = heatmap(corrcoef(var_data));
H.XDisplayLabels = vnames;
H.XDisplayLabels = erase(strrep(vnames,'_',' '),'1');
H.YDisplayLabels = erase(strrep(vnames,'_',' '),'1');