peak = load('../julia/peaks.txt');
M=20;
for m = 1:M
data = importdata(sprintf("../julia/output%d.txt",m),",",2);
output(:,:,m) = data.data;
end

[X,Y] = meshgrid(-3:.01:3,-3:.01:3);
Z = griddata(peak(:,1),peak(:,2),peak(:,3),X,Y);
figure;contourf(X,Y,Z,linspace(min(peak(:,3)), max(peak(:,3)), 100),'LineStyle','none');hold on

outputFile = 'peaks.gif';

for i = 1:size(output,1)
    if exist("p")
        delete(p)
    end
    for m = 1:M
    p(m) = scatter(output(i,3,m),output(i,4,m),20,"red",'filled');
    end
    xlim([-3,3]);
    ylim([-3,3]);
    axis square;
%     pause(0.01)
    title(sprintf('Peaks Function: Iteration %d',i))

    if i==1
        exportgraphics(gca,outputFile);
    else
        exportgraphics(gca,outputFile,'Append',true)
    end

end
