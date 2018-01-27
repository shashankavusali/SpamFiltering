%
% Idx 1 - ham, idx 2 - spam
% class labels - 0 for ham, 1 for spam

%% Variable initialization
ntraindocs = 700;
nwords = 2500;
nclasses = 2;
ntestdocs = 260;
%% Reading data
M = dlmread('ex6DataPrepared/train-features.txt', ' ');
train_labels = dlmread('ex6DataPrepared/train-labels.txt');
sparsemat = sparse(M(:,1), M(:,2),M(:,3),ntraindocs, nwords);
train_data = full(sparsemat);
M = dlmread('ex6DataPrepared/test-features.txt',' ');
sparsemat = sparse(M(:,1), M(:,2), M(:,3));
test_data = full(sparsemat);
test_labels = dlmread('ex6DataPrepared/test-labels.txt');

%% Training
priors = zeros(1,2);
for i = 1: nclasses
    priors(i) = sum(train_labels==(i-1))/ntraindocs;
end

cond_probs = zeros(nclasses, nwords);

for i = 1 : nclasses
    idx = train_labels == (i-1);
    email_for_class = train_data(idx, :);
    cond_probs(i,:) = (sum(email_for_class,1)+1)/(sum(email_for_class(:))+nwords);
end

%% Testing
test_probs = zeros(ntestdocs,2);
for i = 1: nclasses
    test_probs(:,i) = test_data*log(cond_probs(i,:))'+log(priors(i));
end

%% Evaluation
[val, idx] = max(test_probs, [], 2);
idx = idx - 1;
accuracy = sum(idx == test_labels)/ntestdocs;
disp(strcat('Classification Accuracy:',num2str(accuracy)));