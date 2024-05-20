function animateParticles(Particle_Positions,ix1, ix2, labels)
    nDim = (size(Particle_Positions,1)-1)/2;
    nRule = size(Particle_Positions,2);
    nParticle = size(Particle_Positions,3);
    nStep = size(Particle_Positions,4);

    % Need to convert to cell for some reason
    particleCell = num2cell(Particle_Positions);

    gifFile = 'output.gif';
    figure;
    for i = 1:nStep
        line([0,0,10,10,0],[0,10,10,0,0],'Color','k');hold on;

        for j = 1:nRule
            P1 = [particleCell{ix1,j,:,i}];
            P2 = [particleCell{ix2,j,:,i}];
            scatter(P1,P2,'filled');hold on;
        end

        title(sprintf('Iteration %d'), i);
        xlim([-10,20]);
        ylim([-10,20]);
        axis square
        if i==1

            exportgraphics(gca,gifFile);
        else
            exportgraphics(gca,gifFile,'Append',true)
        end
        xlabel(labels{1});
        ylabel(labels{2});
        cla;
    end
end

