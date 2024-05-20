classdef PSO
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
        n = 0;
        J = 0;
    end

    methods
        function obj = PSO(rule,M)
        %UNTITLED4 Construct an instance of this class
        %   Detailed explanation goes here
        obj.parameter_M = M;
        obj.n = length(rule.xq);
        obj.J = rule.j;
        % define P -- matrix of size (2n+1) x J with n rows of c,
        % n rows of sigma, and n rows of w.\
        for m = 1:obj.parameter_M
            obj.particle(m).P = [reshape([rule.cluster.c],obj.n,[]);reshape([rule.cluster.sigma],obj.n,[]);reshape([rule.cluster.w],[],rule.j)];
            obj.particle(m).V = (obj.parameter_Vmax+obj.parameter_Vmax)*rand(2*obj.n+1,rule.j) - obj.parameter_Vmax;
            obj.particle(m).Pbest = obj.particle(m).P;
            obj.particle(m).Hbest = -Inf;
        end

        obj.particle_best.Pbest_g = obj.particle.P;
        end

        function y_hat = calculate_NFS(obj,P,x)
        %METHOD1 Summary of this method goes here
        %   Detailed explanation goes here
            O3 = nan([1,obj.J]);
            for j = 1:obj.J
                mu = gaussian(x,P(1:obj.n,j),P(obj.n+1:2*obj.n,j));
                O3(j) = prod(mu);
            end
            y_hat = f(obj,P(end,:),O3);
        end

        function y_hat = calculate_NFS_improved(obj,P,x)
        %METHOD1 Summary of this method goes here
        %   Detailed explanation goes here
            O3 = nan([1,obj.J]);
            for j = 1:obj.J
                mu = gaussian(x,P(1:obj.n,j),P(obj.n+1:2*obj.n,j));
                O3(j) = prod(mu);
            end
            y_hat = sum(O3.*P(end,:));
        end

        function [H,ACC] = H(obj,y_hat,training_out)
            TP = sum(y_hat(logical(training_out)) == 1);
            TN = sum(y_hat(~logical(training_out)) == 0);
            FN = sum(y_hat(logical(training_out)) == 0);
            FP = sum(y_hat(~logical(training_out)) == 1);
        
            ACC = (TP + TN)/(TP + FP + TN + FN);
            Se = TP/(TP + FN);
            Sp = TN/(FP + TN);

            H = obj.parameter_alpha*ACC + obj.parameter_Beta*Se + obj.parameter_gamma*Sp;
        end

        function [H,ACC] = H_improved(obj,y_hat,training_out)
%             TP_error = sy_hat-training_out
            TP = sum(y_hat(logical(training_out))); % these yhat should all equal 1
            TN = sum(1 - y_hat(logical(~training_out))); % these yhat should all equal 0

            y_hat_true = abs(y_hat(logical(training_out))); % these should all be greater than 0.5
            y_hat_false = abs(y_hat(~logical(training_out))); % these should all be less than 0.5

            TP = sum(y_hat_true(y_hat_true >= 0.5));
            TN = sum(1 - y_hat_false(y_hat_false < 0.5));
            FN = sum(1 - y_hat_true(y_hat_true < 0.5));
            FP = sum(y_hat_false(y_hat_false >= 0.5));

            ACC = (TN + TP)/length(training_out);
            
        
            ACC = (TP + TN)/(TP + FP + TN + FN);
            Se = TP/(TP + FN);
            Sp = TN/(FP + TN);

            H = obj.parameter_alpha*ACC + obj.parameter_Beta*Se + obj.parameter_gamma*Sp;
        end
        
        function obj = update(obj, m)

            % Update V to V(t+1)
            obj.particle(m).V = obj.parameter_W * obj.particle(m).V ...
                + obj.parameter_phi1 .* rand(size(obj.particle(m).P,1),obj.J) .* (obj.particle(m).Pbest - obj.particle(m).P)...
                + obj.parameter_phi2 .* rand(size(obj.particle(m).P,1),obj.J) .* (obj.particle_best.Pbest_g - obj.particle(m).P);
            obj.particle(m).V(obj.particle(m).V > obj.parameter_Vmax) = obj.parameter_Vmax;
            obj.particle(m).V(obj.particle(m).V < -obj.parameter_Vmax) = -obj.parameter_Vmax;

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