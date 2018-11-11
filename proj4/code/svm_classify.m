% Starter code prepared by James Hays for Computer Vision

%This function will train a linear SVM for every category (i.e. one vs all)
%and then use the learned linear classifiers to predict the category of
%every test image. Every test feature will be evaluated with all 15 SVMs
%and the most confident SVM will "win". Confidence, or distance from the
%margin, is W*X + B where '*' is the inner product or dot product and W and
%B are the learned hyperplane parameters.

function predicted_categories = svm_classify(train_image_feats, train_labels, test_image_feats)
% image_feats is an N x d matrix, where d is the dimensionality of the
%  feature representation.
% train_labels is an N x 1 cell array, where each entry is a string
%  indicating the ground truth category for each training image.
% test_image_feats is an M x d matrix, where d is the dimensionality of the
%  feature representation. You can assume M = N unless you've modified the
%  starter code.
% predicted_categories is an M x 1 cell array, where each entry is a string
%  indicating the predicted category for each test image.

%{
Useful functions:
 matching_indices = strcmp(string, cell_array_of_strings)
 
  This can tell you which indices in train_labels match a particular
  category. This is useful for creating the binary labels for each SVM
  training task.

[W B] = vl_svmtrain(features, labels, LAMBDA)
  http://www.vlfeat.org/matlab/vl_svmtrain.html

  This function trains linear svms based on training examples, binary
  labels (-1 or 1), and LAMBDA which regularizes the linear classifier
  by encouraging W to be of small magnitude. LAMBDA is a very important
  parameter! You might need to experiment with a wide range of values for
  LAMBDA, e.g. 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10.

  Matlab has a built in SVM, see 'help svmtrain', which is more general,
  but it obfuscates the learned SVM parameters in the case of the linear
  model. This makes it hard to compute "confidences" which are needed for
  one-vs-all classification.

%}

%unique() is used to get the category list from the observed training
%category list. 'categories' will not be in the same order as in proj4.m,
%because unique() sorts them. This shouldn't really matter, though.
categories = unique(train_labels); 
num_categories = length(categories);


N = size(train_image_feats,1);
d = size(train_image_feats,2);
num_train_data = floor(0.8*N)
indices = randsample(N,N);
train_data = train_image_feats(indices(1:num_train_data),:);
train_data_labels = train_labels(indices(1:num_train_data),1);
hold_data = train_image_feats(indices(num_train_data+1:end),:);
hold_data_labels = train_labels(indices(num_train_data+1:end),1);
W_all = zeros(d,num_categories);
B_all = zeros(1,num_categories);
for indx=1:num_categories
%     labels = double(strcmp(categories(indx), train_data_labels));
    labels = double(strcmp(categories(indx), train_labels));
    idx = (labels == 0);
    labels(idx) = -1;
%     [W, B] = vl_svmtrain(train_data', labels',0.9);
    [W, B] = vl_svmtrain(train_image_feats', labels',0.0009);
    W_all(:,indx) = W;
    B_all(indx) = B;
end
W_a = [W_all' B_all'];
% hold_data = [hold_data ones(size(hold_data,1),1)]';
test_image_feats = [test_image_feats, ones(size(test_image_feats,1),1)]';
% decisions = W_a*hold_data;
decisions = W_a*test_image_feats;
[~, ii] = max(decisions, [], 1);
predicted = categories(ii,1);
% result = double(strcmp(predicted, hold_data_labels));
% result = double(strcmp(predicted, hold_data_labels));
% sum(result)/length(result)
predicted_categories = predicted;
size(predicted_categories)
% COMMENT OUT THE PART BELOW AND UNCOMMENT THE PART ABOVE TO RUN THE CODE
% USED TO TUNE LAMBDA
% test_image_feats = [test_image_feats ones(size(test_image_feats,1),1)]';
% size(W_a)
% size(test_image_feats)
% decisions = W_a*test_image_feats;
% [~, ii] = max(decisions, [], 1);
% predicted_categories = categories(ii,1);
end



