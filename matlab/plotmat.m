function plotmat(matrix, colors)
    for i = 1:size(matrix,2)
        plot(matrix(:,i),'Color',colors(i));hold on;
    end
end