function y = gaussian(x,mu,sigma)
y = exp(-(x-mu).^2./sigma.^2);
end