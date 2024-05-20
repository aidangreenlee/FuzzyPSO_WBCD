c = 5; % initial number of clusters
l = 1;
U = rand([c,size(x,1)]); % sum(U,2) must equal 1, U is an i by k matrix of mu_ik
m = 1.2; % adjustable weight between 1.5 and 3?
v = [];
d = [];
unorm = [];
epsilon = 0.00000001;

while 1
    for i = 1:c
        v(i,:) = sum((U(i,:,l).^m)*x,1) ./ sum(U(i,:,l));
        d(i,:,l) = vecnorm(x-v(i,:),2,2);
    end

    % Calculate I_k
    l = l+1;
    I_k = find(d(:,:,l-1) == 0);
    I_bar_k = setdiff(1:c, I_k);
    if isempty(I_k)
        for i = 1:c
            for k = 1:size(d(:,:,l-1),2)
                U(i,k,l) = 1./sum((d(i,k,l-1)./d(:,k,l-1)).^(2/(m-1)));
            end
        end
    else
        for i = 1:c
            if ismember(i,I_bar_k)
                U(i,:,l) = 0;
            elseif ismember(i,I_k)
                I_k(i,:,l) = I_k(i,:,l)./vecnorm(i,:,l);
            end
        end
    end
    unorm(end+1) = norm(U(:,:,l-1) - U(:,:,l));

    if norm(U(:,:,l-1) - U(:,:,l)) <= epsilon
        break
    end
end

[~,class] = max(U(:,:,end));
% figure
% for i
% for i = 1:c 
%     idx = class == i;
%     scatter(x(idx,1),x(idx,2),[]);hold on
% end


%% Now plot stuff
% % % % for ii = 1:10
% % % %     figure;
% % % %     sgtitle(erase(strrep(vnames(ii),'_',' '),'1'))
% % % %     vindex = setdiff(1:10,ii);
% % % %     for i = 1:length(vindex)
% % % %         subplot(3,3,i)
% % % %         for ic = 1:c
% % % %             idx = class == ic;
% % % %             scatter(x(idx,ii),x(idx,vindex(i)),[]);hold on
% % % %             hold on
% % % %         end
% % % %         ylabel(erase(strrep(vnames(vindex(i)),'_',' '),'1'))
% % % %     end
% % % % end

%%
mu = [];
sigma = [];
d = [];
S = [];
figure;
for ii = 1:10
    subplot(2,5,ii)
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
        plot(x_vals,y_vals,'LineWidth',2);
        hold on
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
