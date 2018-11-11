% Local Feature Stencil Code
% CS 4476 / 6476: Computer Vision, Georgia Tech
% Written by James Hays

% 'features1' and 'features2' are the n x feature dimensionality features
%   from the two images.
% If you want to include geometric verification in this stage, you can add
% the x and y locations of the interest points as additional features.
%
% 'matches' is a k x 2 matrix, where k is the number of matches. The first
%   column is an index in features1, the second column is an index
%   in features2. 
% 'Confidences' is a k x 1 matrix with a real valued confidence for every
%   match.
% 'matches' and 'confidences' can empty, e.g. 0x2 and 0x1.
function [matches, confidences] = match_features(features1, features2)

% This function does not need to be symmetric (e.g. it can produce
% different numbers of matches depending on the order of the arguments).

% To start with, simply implement the "ratio test", equation 4.18 in
% section 4.1.3 of Szeliski. For extra credit you can implement various
% forms of spatial verification of matches.

% Placeholder that you can delete. Random matches and confidences
% num_features = min(size(features1, 1), size(features2,1));
% matches = zeros(num_features, 2);
% matches(:,1) = randperm(num_features); 
% matches(:,2) = randperm(num_features);
% confidences = rand(num_features,1);

num_features_1 = size(features1,1);
num_features_2 = size(features2,1);
count = 1;
for indx = 1:num_features_1
    feature_to_comp = features1(indx,:);
    matrix_feature_to_comp = repmat(feature_to_comp, num_features_2, 1);
    distance = sqrt(sum((matrix_feature_to_comp-features2).^2,2));
    [N1, ind_N1] = min(distance);
    distance(ind_N1) = inf;
    [N2, ~] = min(distance);
    if (N1/N2) <= 0.91
        matches(count,1) = indx;
        matches(count,2) = ind_N1;
        confidences(count) = N1/N2;
        count = count+1;
    end
end
        
    


% Sort the matches so that the most confident onces are at the top of the
% list. You should probably not delete this, so that the evaluation
% functions can be run on the top matches easily.
[confidences, ind] = sort(confidences, 'descend');
matches = matches(ind,:);

end


