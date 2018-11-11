% Fundamental Matrix Stencil Code
% CS 4476 / 6476: Computer Vision, Georgia Tech
% Written by Henry Hu

% Returns the camera center matrix for a given projection matrix

% 'Points_a' is nx2 matrix of 2D coordinate of points on Image A
% 'Points_b' is nx2 matrix of 2D coordinate of points on Image B
% 'F_matrix' is 3x3 fundamental matrix

% Try to implement this function as efficiently as possible. It will be
% called repeatly for part III of the project

function [ F_matrix ] = estimate_fundamental_matrix(Points_a,Points_b)

%%%%%%%%%%%%%%%%
a_row = size(Points_a,1);
b_row = size(Points_b,1);
c_a = sum(Points_a)/(a_row);
c_u_a = c_a(1)
c_v_a = c_a(2)
c_b = sum(Points_b)/(b_row);
c_u_b = c_b(1)
c_v_b = c_b(2)
std_a = std(Points_a(:));
std_b = std(Points_b(:));
s_a = 1/std_a
s_b = 1/std_b

% Points_a = transpose([Points_a ones(size(Points_a,1),1)]);
% Points_b = transpose([Points_b ones(size(Points_b,1),1)]);
% offset_a = [1 0 -c_u_a;0 1 -c_v_a;0 0 1];
% offset_b = [1 0 -c_u_b;0 1 -c_v_b;0 0 1];
% scale_a = [s_a 0 0;0 s_a 0;0 0 1];
% scale_b = [s_b 0 0;0 s_b 0;0 0 1];
% Ta = scale_a*offset_a
% Tb = scale_b*offset_b
% Points_a = Ta*Points_a;
% Points_a = transpose(Points_a(1:end-1,:));
% Points_b = Tb*Points_b;
% Points_b = transpose(Points_b(1:end-1,:));


A = zeros(a_row,9);
for indx = 1:a_row
    A(indx,1) = Points_a(indx,1)*Points_b(indx,1);
    A(indx,2) = Points_a(indx,1)*Points_b(indx,2);
    A(indx,3) = Points_a(indx,1);
    A(indx,4) = Points_a(indx,2)*Points_b(indx,1);
    A(indx,5) = Points_a(indx,2)*Points_b(indx,2);
    A(indx,6) = Points_a(indx,2);
    A(indx,7) = Points_b(indx,1);
    A(indx,8) = Points_b(indx,2);
    A(indx,9) = 1;
end
[U, S, V] = svd(A);
f = V(:,end);
F = reshape(f, [3 3])';
[U, S, V] = svd(F);
S(3,3) = 0;
F_matrix = U*S*V';
% F_matrix = transpose(Tb)*F_matrix*Ta;
%%%%%%%%%%%%%%%%

%This is an intentionally incorrect Fundamental matrix placeholder
% F_matrix = [0  0     -.0004; ...
%             0  0      .0032; ...
%             0 -0.0044 .1034];
%         
end

