function [bboxes, confidences, image_ids] = .... 
    my_hog_run_detector(test_scn_path, w, b, feature_params)
% 'test_scn_path' is a string. This directory contains images which may or
%    may not have faces in them. This function should work for the MIT+CMU
%    test set but also for any other images (e.g. class photos)
% 'w' and 'b' are the linear classifier parameters
% 'feature_params' is a struct, with fields
%   feature_params.template_size (default 36), the number of pixels
%      spanned by each train / test template and
%   feature_params.hog_cell_size (default 6), the number of pixels in each
%      HoG cell. template size should be evenly divisible by hog_cell_size.
%      Smaller HoG cell sizes tend to work better, but they make things
%      slower because the feature dimensionality increases and more
%      importantly the step size of the classifier decreases at test time.

% 'bboxes' is Nx4. N is the number of detections. bboxes(i,:) is
%   [x_min, y_min, x_max, y_max] for detection i. 
%   Remember 'y' is dimension 1 in Matlab!
% 'confidences' is Nx1. confidences(i) is the real valued confidence of
%   detection i.
% 'image_ids' is an Nx1 cell array. image_ids{i} is the image file name
%   for detection i. (not the full path, just 'albert.jpg')

% The placeholder version of this code will return random bounding boxes in
% each test image. It will even do non-maximum suppression on the random
% bounding boxes to give you an example of how to call the function.

% Your actual code should convert each test image to HoG feature space with
% a _single_ call to vl_hog for each scale. Then step over the HoG cells,
% taking groups of cells that are the same size as your learned template,
% and classifying them. If the classification is above some confidence,
% keep the detection and then pass all the detections for an image to
% non-maximum suppression. For your initial debugging, you can operate only
% at a single scale and you can skip calling non-maximum suppression. Err
% on the side of having a low confidence threshold (even less than zero) to
% achieve high enough recall.

test_scenes = dir( fullfile( test_scn_path, '*.jpg' ));

%initialize these as empty and incrementally expand them.
bboxes = [];
confidences = [];
image_ids = {};
num_detect = 1;
num_scales = 8;
scale = 0.8;
window_size = feature_params.template_size/feature_params.hog_cell_size;
for i = 1:length(test_scenes)
    bboxes_im = [];
    confidences_im = [];
    image_ids_im= {}; 
    count = 1;
    fprintf('Detecting faces in %s\n', test_scenes(i).name)
    img = imread( fullfile( test_scn_path, test_scenes(i).name ));
    img = single(img)/255;
    if(size(img,3) > 1)
        img = rgb2gray(img);
    end
    img_size = size(img);
    m = m - rem(m,36);
    n = n - rem(n,36);
    img = imresize(img, [m,n]);
    sliced_image = mat2cell(img,36*ones(1,single(size(img,1)/36)), 36*ones(1,single(size(img,2)/36)));
    num_slice = length(sliced_image); 
    for scale_pow=0:num_scales
        for iter=1:num_slice
            patch = sliced_image{iter};
            my_hog = my_HOG(patch,feature_params.template_size);
            conf = my_hog*w + b;
            if conf > -1.1
                x_min = floor((x_start*feature_params.hog_cell_size - feature_params.hog_cell_size)/(scale^scale_pow));
                y_min = floor((y_start*feature_params.hog_cell_size - feature_params.hog_cell_size)/(scale^scale_pow));
                x_max = x_min + floor(feature_params.template_size/(scale^scale_pow));
                y_max = y_min + floor(feature_params.template_size/(scale^scale_pow));
                bboxes_curr= [x_min, y_min, x_max, y_max];
                bboxes_im = [bboxes_im; bboxes_curr];
                confidences_im = [confidences_im; conf];
                image_ids_im = [image_ids_im;{test_scenes(i).name}];
                num_detect = num_detect + 1;
                count = count + 1;
            end
         end
    end
    end

if size(bboxes_im,1) ~= 0
    [is_max] = non_max_supr_bbox(bboxes_im, confidences_im, size(img));
    confidences = [confidences;confidences_im(is_max,:)];
    bboxes = [bboxes; bboxes_im(is_max,:)];
    image_ids = [image_ids;image_ids_im(is_max,:)];
end
end
