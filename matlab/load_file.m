data = importdata("../julia/outputs/output1.txt",",",2);
output = data.data;
pSize = erase(data.textdata{1}, 'particleSize: ');

parameters = split(data.textdata{2}, ', ');
W = str2double(erase(parameters{1},'W:'));
phi_1 = str2double(erase(parameters{2},'phi_1:'));
phi_2 = str2double(erase(parameters{3},'phi_2:'));

pSize = str2num(pSize);
nDim = (pSize - 1)/2;

overall_fitness = importdata("../julia/outputs/fitness.txt");

names = {'radius','texture','perimeter','area','smoothness','compactness','concavity','concave points',...
    'symmetry','fractal dimension'};
%%W
nCluster = 5;
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
for i = 1:nCluster
    figure;
    %% Plot Position
    subplot(4,3,[1,4])
    plotmat(output(:,pos(:,i)),colors);
    ylim([-10,15])
%     xlabel('Iteration');
    ylabel('Position');
    title('MF Center');
    legend(names,'Location','best','NumColumns',2)

    subplot(4,3,[7,10]);
    plotmat(output(:,pos_std(:,i)),colors);
    ylim([-5,15])
    xlabel('Iteration');
    ylabel('Position');
    title('MF Spread');

    subplot(4,3,[2,5])
    plotmat(output(:,vel(:,i)),colors);
    ylim([-5,5])
%     xlabel('Iteration');
    ylabel('Velocity');
    title('MF Center Velocity');

    subplot(4,3,[8,11]);
    plotmat(output(:,vel_std(:,i)),colors);
    ylim([-5,5])
    xlabel('Iteration');
    ylabel('Velocity');
    title('MF Spread Velocity');

    subplot(4,3,[3,6])
    % plot(output(:, fitness));
    plot(overall_fitness(:,1),'DisplayName','Minimum fitness'); hold on
    plot(overall_fitness(:,2),'DisplayName','Maximum fitness');
    plot(overall_fitness(:,3),'DisplayName','Average fitness');
    plot(overall_fitness(:,4),'DisplayName','Median fitness');
    plot(overall_fitness(:,5),'DisplayName','Global best fitness');
    ylabel('Fitness');
%     xlabel('Iteration')
    ylim([0,1])
    title('Particle Fitness');
    legend('Location','best')
    sgtitle(sprintf('Cluster %d --W:%0.3f \\phi_1:%0.3f \\phi_2:%0.3f',i,W, phi_1, phi_2))

    subplot(4,3,[9,12]);
    plot(output(:,pos_w(:,i)));hold on;
    plot(output(:,vel_w(:,i)));
    xlabel('Iteration');
    ylabel('Output weight');
    legend({'Position', 'Velocity'},'Location','best');


    set(gcf,'Position',[372 147 1191 650])

    figure(MFfig);
    subplot(nCluster,1,i)
    for d = 1:nDim
        plot(x,gaussian(x,output(end,pos(d,i)),output(end,pos_std(d,i))),'Color',colors(d));hold on;
    end
end