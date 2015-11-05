# this takes the simulation from prior results
# and fits a quadratic function of bw to the
# absolute error, and finds the minimizer
load tune_local.out;
data = tune_local;
test = data(:,1) != 1;
data = data(test,:);

y = data(:,3);
x = data(:,2);
n = rows(x);
X = [ones(n,1) x x.^2];
a = ols(y,X);
yhat = X*a;
plot(x, yhat);
bw = -a(2,:)/(2*a(3,:));
bw

figure;
y = data(:,4);
x = data(:,2);
plot(x,y);
n = rows(x);
X = [ones(n,1) x x.^2];
a = ols(y,X);
yhat = X*a;
plot(x, yhat);
bw = -a(2,:)/(2*a(3,:));
bw
