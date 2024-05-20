function [OUT1, OUT2] = getParticlePositions(P)
    OUT1 = nan([2*P.n + 1, P.J, P.parameter_M]);
    OUT2 = nan([2*P.n + 1, P.J, P.parameter_M]);
    for m = 1:P.parameter_M
        OUT1(:,:,m) = P.particle(m).P;
        OUT2(:,:,m) = P.particle(m).V;
    end
end