% Local Feature Stencil Code
% CS 4476 / 6476: Computer Vision, Georgia Tech
% Written by James Hays

% Returns a set of feature descriptors for a given set of interest points. 

% 'image' can be grayscale or color, your choice.
% 'x' and 'y' are nx1 vectors of x and y coordinates of interest points.
%   The local features should be centered at x and y.
% 'feature_width', in pixels, is the local feature width. You can assume
%   that feature_width will be a multiple of 4 (i.e. every cell of your
%   local SIFT-like feature will have an integer width and height).
% If you want to detect and describe features at multiple scales or
% particular orientations you can add input arguments.

% 'features' is the array of computed features. It should have the
%   following size: [length(x) x feature dimensionality] (e.g. 128 for
%   standard SIFT)

function [features] = get_features(image, x, y, feature_width)

% To start with, you might want to simply use normalized patches as your
% local feature. This is very simple to code and works OK. However, to get
% full credit you will need to implement the more effective SIFT descriptor
% (See Szeliski 4.1.2 or the original publications at
% http://www.cs.ubc.ca/~lowe/keypoints/)

% Your implementation does not need to exactly match the SIFT reference.
% Here are the key properties your (baseline) descriptor should have:
%  (1) a 4x4 grid of cells, each feature_width/4. 'cell' in this context
%    nothing to do with the Matlab data structue of cell(). It is simply
%    the terminology used in the feature literature to describe the spatial
%    bins where gradient distributions will be described.
%  (2) each cell should have a histogram of the local distribution of
%    gradients in 8 orientations. Appending these histograms together will
%    give you 4x4 x 8 = 128 dimensions.
%  (3) Each feature vector should be normalized to unit length
%
% You do not need to perform the interpolation in which each gradient
% measurement contributes to multiple orientation bins in multiple cells
% As described in Szeliski, a single gradient measurement creates a
% weighted contribution to the 4 nearest cells and the 2 nearest
% orientation bins within each cell, for 8 total contributions. This type
% of interpolation probably will help, though.

% You do not have to explicitly compute the gradient orientation at each
% pixel (although you are free to do so). You can instead filter with
% oriented filters (e.g. a filter that responds to edges with a specific
% orientation). All of your SIFT-like feature can be constructed entirely
% from filtering fairly quickly in this way.

% You do not need to do the normalize -> threshold -> normalize again
% operation as detailed in Szeliski and the SIFT paper. It can help, though.

% Another simple trick which can help is to raise each element of the final
% feature vector to some power that is less than one.

%Placeholder that you can delete. Empty features.

%features = zeros(size(x,1), 128);
if size(image,3) == 3
    I = rgb2gray(image);
else
    I = image;
end

image_size = size(I);
sigma_weight = 1.5;
offset = feature_width/4;
G_fall_of_function = fspecial('Gaussian', fix(4*sigma_weight+1), sigma_weight);
[image_mag, image_dir] = imgradient(I);
features = [];
for index=1:length(x)
    x_idx = x(index);
    y_idx = y(index);
    w_x_start = x_idx - (feature_width/2) + 1;
    w_x_end = x_idx + feature_width/2;
    w_y_start = y_idx - (feature_width/2) + 1;
    w_y_end = y_idx + feature_width;
    features_vector = [];
    if (w_x_start >= 10) && (w_x_end <= (image_size(2) - 10)) && (w_y_start >= 10) && (w_y_end <= image_size(1) - 10)
        window_mag = image_mag(w_y_start:w_y_end, w_x_start:w_x_end);
        window_dir = image_dir(w_y_start:w_y_end, w_x_start:w_x_end);
        weighted_window_mag = conv2(window_mag, G_fall_of_function, 'same');
        for j_=1:offset
            for i_=1:offset
                cell_mag = weighted_window_mag(j_*offset-offset+1:j_*offset, i_*offset-offset+1:i_*offset);
                cell_dir = window_dir(j_*offset-offset+1:j_*offset, i_*offset-offset+1:i_*offset);
                features_vector = [features_vector compute_HOG(cell_mag, cell_dir)];
            end
        end
        features(index,:) = features_vector;    
    end   
end

features = normr(features);

    function [HOG_] = compute_HOG(Grad_mag, Grad_dir)
       block_size = size(Grad_mag);
       HOG_ = zeros(1,8);
       for i = 1:block_size(1)
           for j = 1:block_size(2)
               mag = Grad_mag(i,j);
               dir = Grad_dir(i,j);
               if (dir > -180) && (dir < -135)
                   HOG_(1) = HOG_(1)+mag;
               elseif (dir == -135)
                   HOG_(1) = HOG_(1) + mag/2;
                   HOG_(2) = HOG_(2) + mag/2;
               elseif (dir > -135) && (dir < -90)
                   HOG_(2) = HOG_(2) + mag;
               elseif (dir == -90)
                   HOG_(2) = HOG_(2) + mag/2;
                   HOG_(3) = HOG_(3) + mag/2;
               elseif (dir > -90) && (dir < -45)
                   HOG_(3) = HOG_(3) + mag;
               elseif (dir == -45)
                   HOG_(3) = HOG_(3) + mag/2;
                   HOG_(4) = HOG_(4) + mag/2;
               elseif (dir > -45) && (dir < 0)
                   HOG_(4) = HOG_(4) + mag;
               elseif (dir == 0)
                   HOG_(4) = HOG_(4) + mag/2;
                   HOG_(5) = HOG_(5) + mag/2;
               elseif (dir > 0) && ( dir < 45)
                   HOG_(5) = HOG_(5) + mag;
               elseif (dir == 45)
                   HOG_(5) = HOG_(5) + mag/2;
                   HOG_(6) = HOG_(6) + mag/2;
               elseif (dir > 45) && (dir < 90)
                   HOG_(6) = HOG_(6) + mag;
               elseif (dir == 90)
                   HOG_(6) = HOG_(6) + mag/2;
                   HOG_(7) = HOG_(7) + mag/2;
               elseif (dir > 90) && (dir < 135)
                   HOG_(7) = HOG_(7) + mag;
               elseif (dir == 135)
                   HOG_(7) = HOG_(7) + mag/2;
                   HOG_(8) = HOG_(8) + mag/2;
               elseif (dir > 135) && (dir < 180)
                   HOG_(8) = HOG_(8) + mag;
               elseif (dir == -180) || (dir == 180)
                   HOG_(8) = HOG_(8) + mag/2;
                   HOG_(1) = HOG_(1) + mag/2;
               end
           end
       end    
    end

end








