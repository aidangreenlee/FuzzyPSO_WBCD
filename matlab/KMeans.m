% calculate point classes and cluster centers
% c = 3;
[class,centers] = kmeans(x,c);
%% Now plot stuff
for ii = 1:10
%     figure;
    sgtitle(erase(strrep(vnames(ii),'_',' '),'1'))
    vindex = setdiff(1:10,ii);
    for i = 1:length(vindex)
%         subplot(3,3,i)
        for ic = 1:c
            idx = class == ic;
%             scatter(x(idx,ii),x(idx,vindex(i)),[]);hold on
%             hold on
        end
        ylabel(erase(strrep(vnames(vindex(i)),'_',' '),'1'))
    end
end
%%
mu = [];
sigma = [];
d = [];
S = [];
% figure;
for ii = 1:10
%     subplot(2,5,ii)
    for ic = 1:c
        idx = class == ic;
        if (~any(idx))
            mu(ii,ic) = nan;
            sigma(ii,ic) = nan;
            d(ic) = nan;
            S(ic) = nan;
            continue
        end
        x_vals = linspace(min(x(idx,ii)),max(x(idx,ii)),1000);
        MF = fitdist(x(idx,ii),"normal");
        mu(ii,ic) = MF.mu;
        sigma(ii,ic) = MF.sigma;
        d(ic) = mean(var_diagnosis(idx));
        S(ic) = sum(idx);
        y_vals = MF.pdf(x_vals);
        y_vals = y_vals./max(y_vals(~isinf(y_vals)));
%         [mu(ii,ic),sigma(ii,ic)] = normfit(x(idx,ii));
%         y_vals = normpdf(x_vals,mu(ii,ic),sigma(ii,ic));
%         plot(x_vals,y_vals,'LineWidth',2);
%         hold on
    end
    title(erase(strrep(vnames{ii},'_',' '),'1'))
    grid on;
end

mu(:,isnan(d)) = [];
S(isnan(d)) = [];
d(isnan(d)) = [];
r = rule(mu(:,1),d(1),0,10);
r.parameter_sigma0 = 2.5;
r.cluster(1).S = S(1);
for ic = 2:length(d)
    r = r.add_rule(mu(:,ic),d(ic),S(ic));
end
