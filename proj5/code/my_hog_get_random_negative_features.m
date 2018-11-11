function my_hog_features_neg = my_hog_get_random_negative_features(non_face_scn_path, feature_params, num_samples)
% 'non_face_scn_path' is a string. This directory contains many images
%   which have no faces in them.
% 'feature_params' is a struct, with fields
%   feature_params.template_size (default 36), the number of pixels
%      spanned by each train / test template and
%   feature_params.hog_cell_size (default 6), the number of pixels in each
%      HoG cell. template size should be evenly divisible by hog_cell_size.
%      Smaller HoG cell sizes tend to work better, but they make things
%      slower because the feature dimensionality increases and more
%      importantly the step size of the classifier decreases at test time.
% 'num_samples' is the number of random negatives to be mined, it's not
%   important for the function to find exactly 'num_samples' non-face
%   features, e.g. you might try to sample some number from each image, but
%   some images might be too small to find enough.

% 'features_neg' is N by D matrix where N is the number of non-faces and D
% is the template dimensionality, which would be
%   (feature_params.template_size / feature_params.hog_cell_size)^2 * 31
% if you're using the default vl_hog parameters

% Useful functions:
% vl_hog, HOG = VL_HOG(IM, CELLSIZE)
%  http://www.vlfeat.org/matlab/vl_hog.html  (API)
%  http://www.vlfeat.org/overview/hog.html   (Tutorial)
% rgb2gray

image_files = dir( fullfile( non_face_scn_path, '*.jpg' ));
num_images = length(image_files);
% placeholder to be deleted. 100 random features.
% features_neg = rand(100, (feature_params.template_size / feature_params.hog_cell_size)^2 * 31);
my_hog_features_neg = [];

for indx=1:num_images
    fileName = image_files(indx).name;
    im = im2single(imread(fullfile( non_face_scn_path, fileName)));
    if size(im,3)==3
        im = rgb2gray(im);
    end
    [m,n] = size(im);
    for i=1:50
        im_patch = im(randi(m-feature_params.template_size+1)+(0:feature_params.template_size-1), randi(n-feature_params.template_size+1)+(0:feature_params.template_size-1));
        my_hog = my_HOG(im_patch, feature_params.template_size);
        my_hog_features_neg = [my_hog_features_neg; my_hog];
    end
    
end