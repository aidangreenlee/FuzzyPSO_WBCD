M=20;

names = {'radius','texture','perimeter','area','smoothness','compactness','concavity','concave points',...
    'symmetry','fractal dimension'};

for m = 1:M
data = importdata(sprintf("../julia/outputs/output%d.txt",m),",",2);
output(:,:,m) = data.data;
end

parameters = split(data.textdata{2}, ', ');
W = str2double(erase(parameters{1},'W:'));
phi_1 = str2double(erase(parameters{2},'phi_1:'));
phi_2 = str2double(erase(parameters{3},'phi_2:'));

figure;
line([0,0,10,10,0],[0,10,10,0,0],'Color','k'); hold on

v = VideoWriter('outputVideo.mp4','MPEG-4');
v.FrameRate = 30;
frame = {};

for i = 1:size(output,1)
    if exist("p")
        delete(p)
    end
    xindex = 4;
    yindex = 5;
    p(1) = scatter(output(i,xindex+3,1),output(i,yindex+3,1),20,"blue",'filled');hold on;
    for m = 2:M
    p(m) = scatter(output(i,xindex+3,m),output(i,yindex+3,m),20,"red",'filled');hold on;
    end
    xlim([-10,20]);
    ylim([-10,20]);
    xlabel(names(xindex));
    ylabel(names(yindex));
    axis square;
%     pause(0.01)
    title(sprintf('PSO: Iteration %d\nW=%0.2f, \\phi_1=%0.2f, \\phi_2=%0.2f',i,W,phi_1,phi_2))

%     if i==1
%         exportgraphics(gca,outputFile);
%     else
%         exportgraphics(gca,outputFile,'Append',true)
%     end
    frame{end+1} = getframe(gcf);
%     pause(0.1)
end

open(v)
for f = 1:length(frame)
    writeVideo(v,frame{f});
end
close(v);
