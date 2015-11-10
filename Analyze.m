# this takes the simulation from prior results
# and fits a quadratic function of bw to the
# absolute error, and finds the minimizer
load tune_from_prior.out;
data = tune_from_prior;
#load tune_local.out;
#data = tune_local;
test = data(:,1) != 1;
data = data(test,:);
test = data < 100;
data = test.*data + 100*(1-test);

bws = data(:,2);
bws = unique(bws);
scores = zeros(rows(bws),4);
for whichdep = 1:4
        for i = 1:rows(bws)
                test = data(:,2) == bws(i,:);
                scores(i,whichdep) = mean(data(test,whichdep+2));
        endfor
endfor
ind = 1:rows(bws);
disp([ind' bws scores])
plot(bws, scores(:,1));
figure;
plot(bws, scores(:,2));
figure;
plot(bws, scores(:,3));
figure;
plot(bws, scores(:,4));
