classdef rule
    %UNTITLED2 Summary of this class goes here
    %   Detailed explanation goes here

    properties
        cluster = struct('c',[],'w',[],'sigma',[],'S',1)
        F = [];
        j = 0;

        % Store current training pattern
        xq = [];
        dq = [];
        dmin = [];
        dmax = [];
        % Parameter settings for SCNF classifier:
        parameter_sigma0 = .25;
        parameter_rho = .1;
        parameter_Kappa = 1.1;
        parameter_tau = 0.001;
    end

    methods
        function obj = rule(x,d,dmin,dmax)
            %UNTITLED2 Construct an instance of this class
            %   Detailed explanation goes here
            obj.cluster(1).c = x;
            obj.j = 1;
            obj.cluster(1).sigma = repmat(obj.parameter_sigma0,size(x));
            obj.cluster(1).w = d;
            obj.dmin = dmin;
            obj.dmax = dmax;
            obj.xq = x;
            obj.dq = d;
        end


        function obj = similarity(obj,varargin)
            if nargin == 3
                obj.xq = varargin{1};
                obj.dq = varargin{2};
            end
            % calculate equation 6
            for k = 1:length(obj.cluster)
                obj.F(k) = obj.F_C(obj.cluster(k).c,obj.cluster(k).sigma);
            end
        end

        function F = F_C(obj,c,sigma)
            F = prod(gaussian(obj.xq,c,sigma));
        end

        function [input_test, output_test] = verify_conditions(obj)
            input_test = obj.F >= obj.parameter_rho; % input similarity test
            output_test = abs(obj.dq-[obj.cluster.w]) <= obj.parameter_tau*(obj.dmax - obj.dmin); % output similarity
        end
        %
        function obj = initialize_rule(obj)
            obj.j = obj.j + 1;
            obj.cluster(obj.j).c = obj.xq; % Equation 10 ck = x_sup(q)
            obj.cluster(obj.j).sigma(:) = repmat(obj.parameter_sigma0,size(obj.xq));
            obj.cluster(obj.j).w = obj.dq;
            obj.cluster(obj.j).S = 1;
        end

        function obj = add_rule(obj,x,d,S)
            obj.j = obj.j + 1;
            obj.cluster(obj.j).c = x;
            obj.cluster(obj.j).w = d;
            obj.cluster(obj.j).sigma = repmat(obj.parameter_sigma0, size(x));
            obj.cluster(obj.j).S = S;
        end

        function obj = update_cluster(obj,v)
            obj.cluster(v).c = (obj.cluster(v).S*obj.cluster(v).c+obj.xq)./(obj.cluster(v).S + 1);
            obj.cluster(v).sigma = repmat(obj.parameter_Kappa*obj.parameter_sigma0,size(obj.cluster(v).sigma));
            obj.cluster(v).w = (obj.cluster(v).S*obj.cluster(v).w+obj.dq)./(obj.cluster(v).S + 1);
            obj.cluster(v).S = obj.cluster(v).S + 1;
        end
    end
end