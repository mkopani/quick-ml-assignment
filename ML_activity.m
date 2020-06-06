% ML Assignment %

clear all

%% Load data
load('mnist.mat')

%% Generate matrices
idx = trainY == 4 | trainY == 9;
A = double(trainX(idx,:));
b = double(trainY(idx));

idx = testY == 4 | testY == 9;
Atest = double(testX(idx,:));
btest = double(testY(idx));

%% Change labels
b_backup = b;

idx = b == 4;
b(idx) = 1;

idx = b == 9;
b(idx) = -1;

idx = btest == 4;
btest(idx) = 1;

idx = btest == 9;
btest(idx) = -1;

%% De-biasing & normalizing
[m, n] = size(A);

% Remove bias
Amean = mean(A, 1);
A = A - ones(m, 1)*Amean;

% Remove variance
Astd = std(A, 1);
A = A ./ max(ones(m,1)*Astd, 1);

[mtest, ntest] = size(Atest);

Atest = Atest - ones(mtest, 1)*Amean; % remove bias (test)
Atest = Atest ./ (ones(mtest,1)*Astd); % remove variance (test)
Atest(isnan(Atest))=0; % replace NaNs with 0's

%% Linear regression
b = b';
btest = btest';

xLS = A\b;

loss = norm(A*xLS-b)^2;

missclassRateTrain = classifier(A, b, xLS);
missclassRateTest = classifier(Atest, btest, xLS);

table(missclassRateTrain, missclassRateTest, 'VariableNames', {'Train', 'Test'})

%% Readjust labels for logistic regression
b = (b+1)/2;  % readjust b to 0,1 labels rather than -1,1
btest = (btest+1)/2;

%% Gradient descent
xk = zeros(n, 1);
alpha = 1/m;

results = [];

dk = @(x) -grad(A, b, x);

counter = 0;

while norm(grad(A, b, xk)) > eps
    results = [results xk];
    xk = xk + alpha*dk(xk);
        
    counter = counter + 1;
    
    if counter >= 1000
        break
    end
end

% Compute losses for each iteration
losses = [];

for i = 1:1000
    losses = [losses; logloss(A, b, results(:,i))];
end

%% Plot & misclassification rates for Gradient Descent
close all

plot(1:1000, losses, 'linewidth', 2);
xlabel('iterations', 'fontsize', 13);
ylabel('loss', 'fontsize', 13);
title('Loss for gradient descent with constant step size', 'fontsize', 18');

missclassRateTrain_GD = classifier3(A, b, results(:,end));
missclassRateTest_GD = classifier3(Atest, btest, results(:,end));

table(missclassRateTrain_GD, missclassRateTest_GD, 'VariableNames', {'Train', 'Test'})

%% Plot & misclassification rates for Gradient Descent with Linesearch
s = 1;
alpha = 0.5;
beta = 0.5;

x0 = zeros(n, 1);
xk = x0;


dk = @(x) -grad(A, b, x);
f = @(x) loglikelihood(A, b, x);

results2 = [];

counter = 0;

while norm(grad(A, b, xk)) > eps
    
    tk = s;
    results2 = [results2 xk];
    
    while (f(xk) - f(xk + tk*dk(xk))) < alpha*tk*grad(A, b, xk)'*dk(xk)
        tk = beta*tk;
    end
    
    xk = xk + tk*dk(xk);
    
    counter = counter + 1;
    
    if counter >= 1000
       break
    end
    
end

%% Compute losses for each iteration
losses2 = [];

for i = 1:1000
    losses2 = [losses2; logloss(A, b, results2(:,i))];
end

%% Plot 
close all

plot(1:1000, losses2, 'linewidth', 2);
xlabel('iterations', 'fontsize', 13);
ylabel('loss', 'fontsize', 13);
title('Loss for gradient descent with backtracking linesearch', 'fontsize', 18');

missclassRateTrain_GDLS = classifier3(A, b, results2(:, end));
missclassRateTest_GDLS = classifier3(Atest, btest, results2(:, end));

table(missclassRateTrain_GDLS, missclassRateTest_GDLS, 'VariableNames', {'Train', 'Test'})

