classdef PSO_test
    %UNTITLED4 Summary of this class goes here
    %   Detailed explanation goes here

    properties
        particle = struct('V',[],'P',[],'Pbest',[],'H',-inf,'Hbest',-inf)
        particle_best = struct('H',-inf,'Pbest_g',[]);
        parameter_Vmax = 4;
        parameter_phi1 = 2;
        parameter_phi2 = 2;
        parameter_alpha = .6;
        parameter_Beta = .2;
        parameter_gamma = .2;
        parameter_W = .8;
        parameter_M = 20;
        m = 1;
        n = 0.5;
        J = 1;
    end

    methods
        function obj = PSO_test(M)
        %UNTITLED4 Construct an instance of this class
        %   Detailed explanation goes here
        obj.parameter_M = M;
        % define P -- matrix of size (2n+1) x J with n rows of c,
        % n rows of sigma, and n rows of w.\
        for m = 1:obj.parameter_M
            obj.particle(m).P = rand(2,1)*6-3; % random number between -3 and 3
            obj.particle(m).V = (obj.parameter_Vmax+obj.parameter_Vmax)*rand(2,1) - obj.parameter_Vmax;
            obj.particle(m).Pbest = obj.particle(m).P;
            obj.particle(m).Hbest = -Inf;
        end

        obj.particle_best.Pbest_g = obj.particle.P;
        end

        function H = H(obj,m)
            H = 8 + peaks(obj.particle(m).P(1),obj.particle(m).P(2));
        end
        
        function obj = update(obj, m)

            % Update V to V(t+1)
            obj.particle(m).V = obj.parameter_W * obj.particle(m).V ...
                + obj.parameter_phi1 .* rand(size(obj.particle(m).P,1),1) .* (obj.particle(m).Pbest - obj.particle(m).P)...
                + obj.parameter_phi2 .* rand(size(obj.particle(m).P,1),1) .* (obj.particle_best.Pbest_g - obj.particle(m).P);
%             obj.particle(m).V(obj.particle(m).V > obj.parameter_Vmax) = obj.parameter_Vmax;
%             obj.particle(m).V(obj.particle(m).V < -obj.parameter_Vmax) = -obj.parameter_Vmax;

            % Update Pbest and Pbest_g
            if obj.particle(m).H  >= obj.particle(m).Hbest
                obj.particle(m).Hbest = obj.particle(m).H;
                obj.particle(m).Pbest = obj.particle(m).P;
            else
                obj.particle(m).Pbest = obj.particle(m).Pbest;
            end

            if obj.particle(m).Hbest > obj.particle_best.H(end)
                obj.particle_best.Pbest_g = obj.particle(m).Pbest;
                obj.particle_best.H(end+1) = obj.particle(m).Hbest;
            end

            % Update P to P(t+1)
            obj.particle(m).P = obj.particle(m).P + obj.particle(m).V;
%             tmp = obj.particle(m).P(obj.n+1:2*obj.n,:);
%             tmp(tmp > 5) = 5;
%             obj.particle(m).P(obj.n+1:2*obj.n,:) = tmp;
%             obj.particle(m).P(obj.n+1:2*obj.n,:) = min(abs(obj.particle(m).P(obj.n+1:2*obj.n,:)),1).*sign(obj.particle(m).P(obj.n+1:2*obj.n,:));

        end
    end

    methods (Access = private)
        function y = f(obj,w,O)
            if sum((w.*O)) < .5
                y = 0;
            else
                y = 1;
            end
        end
    end
end