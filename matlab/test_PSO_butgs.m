% Now do particle swarm
P = PSO_test(20);
P.parameter_Vmax = 3;
% P.parameter_Vmax = 2;s
P.parameter_alpha = 0.2;
P.parameter_Beta = 0.4;
P.parameter_gamma = 0.4;
P.parameter_W = 0.8;
P.parameter_phi1 = 1.2;
P.parameter_phi2 = 1.4;
y_hat = nan(20,1);
c = 0; % count number of loops
H = [];
Hmin = [];
Hmax = [];
Hmean = [];
Hmedian = [];
Hq1 = [];
Hq3 = [];
tmp1 = [];
tmp2 = [];
% while P.particle_best.H(end) < .98

clear Particle_Positions
clear Particle_Velocities
[Particle_Positions(:,:,:,1), Particle_Velocities(:,:,:,1)] = getParticlePositions(P);

W = linspace(3,0.1,400);
Phi1 = linspace(4,1,400);
Phi2 = linspace(2,5,400);

while c <= 101% || length(unique(H(end-100:end))) ~= 1
%     if c >= 500 || any([P.particle.Hbest] > .98)
%         break
%     end
%     if (c+1) >= 400
%         P.parameter_W = .1;
% %         P.parameter_phi1 = 1;
% %         P.parameter_phi2 = 4;
%     else
%         P.parameter_W = W(c+1);
% %         P.parameter_phi1 = Phi1(c+1);
% %         P.parameter_phi2 = Phi2(c+1);
%     end
    for m = 1:P.parameter_M

        P.particle(m).H = P.H(m);
        P = P.update(m);
    end
%     P.particle_best.H(end+1) = max(H);
%     idx = find(H == P.particle_best.H(end),1, 'last');
%     P.particle_best.Pbest_g = P.particle(idx).P;
    disp(P.particle_best.H(end))
    c = c+1;
end