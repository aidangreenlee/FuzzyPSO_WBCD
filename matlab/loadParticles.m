function output = loadParticles(rundir, particle_numbers)
if nargin < 1
    rundir = '../julia/';
    particle_numbers = 1:20;
elseif nargin < 2
    particle_numbers = 1:20;
end
for m = particle_numbers
data = importdata(fullfile(rundir,sprintf("output%d.txt",m)),",",2);
output(:,:,m) = data.data;
end
end