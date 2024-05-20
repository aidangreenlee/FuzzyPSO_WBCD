function y = calculate_NFS(x,mu,sigma, a0, a)
    % x is n by 1 array
    % mu is n by J matrix
    % sigma is n by J matrix
    % a0 is a 1 by J array
    % a is a n by J matrix
    O1 = x;
    O2 = exp(-(x-mu)./sigma);
    O3 = prod(O2,1);
    O4 = O3.*(a0 + sum(a.*x));
    t = sum(O4)./sum(O3);
    y = 1/(1+exp(-t));
end