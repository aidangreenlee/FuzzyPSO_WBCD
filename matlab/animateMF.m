for i = 1:20
    data = importdata(sprintf("../julia/temp/output%d.txt",i),",",2);
    output(:,:,i) = data.data;
end
pSize = erase(data.textdata{1}, 'particleSize: ');

parameters = split(data.textdata{2}, ', ');
W = str2double(erase(parameters{1},'W:'));
phi_1 = str2double(erase(parameters{2},'phi_1:'));
phi_2 = str2double(erase(parameters{3},'phi_2:'));

pSize = str2num(pSize);
nDim = (pSize - 1)/2;

names = {'radius','texture','perimeter','area','smoothness','compactness','concavity','concave points',...
    'symmetry','fractal dimension'};
%%W
nCluster = 4;
index = 1;
fitness = 2;

position = reshape((fitness+1):(fitness+pSize*nCluster),pSize, nCluster);
pos = position(1:nDim,:);
pos_std = position(nDim + 1:2*nDim,:);
pos_w = position(end,:);

velocity = reshape((position(end) + 1):size(output,2),pSize,nCluster);
vel = velocity(1:nDim,:);
vel_std = velocity(nDim + 1:2*nDim,:);
vel_w = velocity(end,:);

%%
colors = ["#008aff",
    "#79de68",
    "#9c4cd7",
    "#f3a839",
    "#9a9d9b",
    "#de3898",
    "#2f1231",
    "#e3711b",
    "#b40000",
    "#29721d"];



MFfig = figure;
x = 0:.01:10;
clusternum = 1
xnum = 1;
ynum = 7;
for i = 1:size(output,1)
    figure(MFfig);
    cla
    %     subplot(1,1,i)
    %     for d = 1:nDim
    %         plot(x,gaussian(x,output(i,pos(d,clusternum)),output(i,pos_std(d,clusternum))),'Color',colors(d),'LineWidth',2);hold on;
    XX = {output(i,pos(xnum,clusternum),:)};
    YY = {output(i,pos(ynum,clusternum),:)};
%     hold off;
    line([0,0,10,10,0],[0,10,10,0,0],'Color','red');hold on
    plot(squeeze(XX{:}),squeeze(YY{:}),'k.');hold on
    grid on;
    axis square
    %         legend(names,'Location','bestoutside')
    xlabel(names{xnum},'FontSize',12);
    ylabel(names{ynum},'FontSize',12);
    xlim([-5,15]);
    ylim([-5,15]);
    %     end
    pause(0.01)
    frames{i} = getframe(gca);
    hold off
end

v = VideoWriter('Video')
open(v);
for i = 1:length(frames)
    writeVideo(v, frames{i});
end
close(v);