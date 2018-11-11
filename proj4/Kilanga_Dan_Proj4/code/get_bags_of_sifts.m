% Starter code prepared by James Hays for Computer Vision

%This feature representation is described in the handout, lecture
%materials, and Szeliski chapter 14.

function image_feats = get_bags_of_sifts(image_paths)
% image_paths is an N x 1 cell array of strings where each string is an
% image path on the file system.

% This function assumes that 'vocab.mat' exists and contains an N x 128
% matrix 'vocab' where each row is a kmeans centroid or visual word. This
% matrix is saved to disk rather than passed in a parameter to avoid
% recomputing the vocabulary every run.

% image_feats is an N x d matrix, where d is the dimensionality of the
% feature representation. In this case, d will equal the number of clusters
% or equivalently the number of entries in each image's histogram
% ('vocab_size') below.

% You will want to construct SIFT features here in the same way you
% did in build_vocabulary.m (except for possibly changing the sampling
% rate) and then assign each local feature to its nearest cluster center
% and build a histogram indicating how many times each cluster was used.
% Don't forget to normalize the histogram, or else a larger image with more
% SIFT features will look very different from a smaller version of the same
% image.

%{
Useful functions:
[locations, SIFT_features] = vl_dsift(img) 
 http://www.vlfeat.org/matlab/vl_dsift.html
 locations is a 2 x n list list of locations, which can be used for extra
  credit if you are constructing a "spatial pyramid".
 SIFT_features is a 128 x N matrix of SIFT features
  note: there are step, bin size, and smoothing parameters you can
  manipulate for vl_dsift(). We recommend debugging with the 'fast'
  parameter. This approximate version of SIFT is about 20 times faster to
  compute. Also, be sure not to use the default value of step size. It will
  be very slow and you'll see relatively little performance gain from
  extremely dense sampling. You are welcome to use your own SIFT feature
  code! It will probably be slower, though.

D = vl_alldist2(X,Y) 
   http://www.vlfeat.org/matlab/vl_alldist2.html
    returns the pairwise distance matrix D of the columns of X and Y. 
    D(i,j) = sum (X(:,i) - Y(:,j)).^2
    Note that vl_feat represents points as columns vs this code (and Matlab
    in general) represents points as rows. So you probably want to use the
    transpose operator '  You can use this to figure out the closest
    cluster center for every SIFT feature. You could easily code this
    yourself, but vl_alldist2 tends to be much faster.

Or:

For speed, you might want to play with a KD-tree algorithm (we found it
reduced computation time modestly.) vl_feat includes functions for building
and using KD-trees.
 http://www.vlfeat.org/matlab/vl_kdtreebuild.html

%}

load('vocab.mat')
vocab_size = size(vocab, 1);
N = size(image_paths,1);
Step = 8;
binSize = 8;
max_level = 2;
feat_dim = single(vocab_size*(1/3)*(4^(max_level+1)-1));
image_feats = zeros(N, feat_dim);
level_hist = zeros(N, vocab_size, 21);
hist_count = 1;
for indx=1:N
    img = imread(image_paths{indx,1});
    if size(img,3) == 3
        img = rgb2gray(img);
    end
    img = single(double(img));
    img_size = size(img);
   [locations, SIFT_features] = vl_dsift(img, 'Step', Step, 'Size', binSize, 'fast');
   % Level 0 computation
    distances = vl_alldist2(vocab', single(SIFT_features), 'CHI2');
    [~,ii0] = min(distances, [], 1);
    [u0, ~, uidx0] = unique(ii0);
    counts = accumarray(uidx0,1);
    level_hist(indx,u0,hist_count) = level_hist(indx,u0,hist_count) + counts';
    hist_count = hist_count + 1;
    % Level 1 computation
    x_step1 = floor(img_size(2)/2);
    y_step1 = floor(img_size(1)/2);
    x_ind_s = 1;
    y_ind_s = 1;
    for y_ind=y_step1:y_step1:img_size(1)-1
        for x_ind=x_step1:x_step1:img_size(2)-1
             admissible = double(sum((locations < repmat([x_ind;y_ind],1,size(locations,2))) & (locations > repmat([x_ind_s;y_ind_s],1,size(locations,2))), 1));
             index = admissible == 2;
             distances11 = vl_alldist2(vocab', single(SIFT_features(:,index)),'CHI2');
             [~,ii1] = min(distances11, [], 1);
             [u1, ~, uidx1] = unique(ii1);
             counts11 = accumarray(uidx1,1);
             level_hist(indx,u1, hist_count) = level_hist(indx,u1, hist_count) + counts11';
             hist_count = hist_count + 1;
             x_ind_s = x_ind_s + x_step1-1;
        end
        x_ind_s = 1;
        y_ind_s = y_ind_s + y_step1-1;
    end
    % Level 2 computation
    x_step1 = floor(img_size(2)/4);
    y_step1 = floor(img_size(1)/4);
    x_ind_s = 1;
    y_ind_s = 1;
    for y_ind=y_step1:y_step1:img_size(1)-8
        for x_ind=x_step1:x_step1:img_size(2)-8
             admissible = double(sum((locations < repmat([x_ind;y_ind],1,size(locations,2))) & (locations > repmat([x_ind_s;y_ind_s],1,size(locations,2))), 1));
             index = admissible == 2;
             distances11 = vl_alldist2(vocab', single(SIFT_features(:,index)), 'CHI2');
             [~,ii1] = min(distances11, [], 1);
             [u1, ~, uidx1] = unique(ii1);
             counts11 = accumarray(uidx1,1);
             level_hist(indx,u1, hist_count) = level_hist(indx,u1, hist_count) + counts11';
             hist_count = hist_count + 1;
             x_ind_s = x_ind_s + x_step1-1;
        end
        x_ind_s = 1;
        y_ind_s = y_ind_s + y_step1-1;
    end
    hist_count = 1;
end
s_idx = 1;
step_idx = vocab_size;
e_idx = step_idx;
for hist_count=1:21
    if hist_count == 1
        image_feats(:,s_idx:e_idx) = (1/4)*(level_hist(:,:,hist_count) - sum(level_hist(:,:,2:5),3));
    elseif hist_count == 2 
        image_feats(:,s_idx:e_idx) = (1/2)*(level_hist(:,:,hist_count)- sum(level_hist(:,:,[6,7,10,11]),3));
    elseif hist_count == 3
        image_feats(:,s_idx:e_idx) = (1/2)*(level_hist(:,:,hist_count)- sum(level_hist(:,:,[8,9,12,13]),3));
    elseif hist_count == 4
        image_feats(:,s_idx:e_idx) = (1/2)*(level_hist(:,:,hist_count)- sum(level_hist(:,:,[14,15,18,19]),3));
    elseif hist_count == 5
        image_feats(:,s_idx:e_idx) = (1/2)*(level_hist(:,:,hist_count)- sum(level_hist(:,:,[16,17,20,21]),3));
    elseif hist_count >= 6 && hist_count <= 21
        image_feats(:,s_idx:e_idx) = level_hist(:,:,hist_count);
    end
    s_idx = s_idx + step_idx;
    e_idx = e_idx + step_idx;
end
image_feats = normr(image_feats);
end

% load('vocab.mat')
% vocab_size = size(vocab, 1);
% N = size(image_paths,1);
% Step = 8;
% binSize = 8;
% image_feats = zeros(N, vocab_size);
% for indx=1:N
%     img = imread(image_paths{indx,1});
%     if size(img,3) == 3
%         img = rgb2gray(img);
%     end
%     img = single(double(img));
%    [~, SIFT_features] = vl_dsift(img, 'Step', Step, 'Size', binSize, 'fast');
%    % Level 0 computation
%    distances = vl_alldist2(vocab', single(SIFT_features), 'CHI2');
%    [~,ii] = min(distances, [], 1);
%    [u, ~, uidx] = unique(ii);
%    counts = accumarray(uidx,1);
%    image_feats(indx,u) = image_feats(indx,u) + counts';
% end
% image_feats = normr(image_feats);
% end




