function plotParticle(Particle_Positions, ipart)
    particleCell = num2cell(Particle_Positions);
    nPSOvars = size(Particle_Positions,1);
    nRule = size(Particle_Positions,2);

    for irule = 1:nRule
%     f = figure;
    for i = 1:(nPSOvars-1)/2
        P = [particleCell{i,irule,ipart,:}];
        plot(P);hold on;
    end
    end
end