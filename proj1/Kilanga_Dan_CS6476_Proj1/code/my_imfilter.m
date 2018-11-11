function output = my_imfilter(image, filter)
% This function is intended to behave like the built in function imfilter()
% See 'help imfilter' or 'help conv2'. While terms like "filtering" and
% "convolution" might be used interchangeably, and they are indeed nearly
% the same thing, there is a difference:
% from 'help filter2'
%    2-D correlation is related to 2-D convolution by a 180 degree rotation
%    of the filter matrix.

% Your function should work for color images. Simply filter each color
% channel independently.

% Your function should work for filters of any width and height
% combination, as long as the width and height are odd (e.g. 1, 7, 9). This
% restriction makes it unambigious which pixel in the filter is the center
% pixel.

% Boundary handling can be tricky. The filter can't be centered on pixels
% at the image boundary without parts of the filter being out of bounds. If
% you look at 'help conv2' and 'help imfilter' you see that they have
% several options to deal with boundaries. You should simply recreate the
% default behavior of imfilter -- pad the input image with zeros, and
% return a filtered image which matches the input resolution. A better
% approach is to mirror the image content over the boundaries for padding.

% % Uncomment if you want to simply call imfilter so you can see the desired
% % behavior. When you write your actual solution, you can't use imfilter,
% % filter2, conv2, etc. Simply loop over all the pixels and do the actual
% % computation. It might be slow.
% output = imfilter(image, filter);


%%%%%%%%%%%%%%%%
% Your code here

size_image = size(image);
size_filter = size(filter);
filtered_image = zeros(size(image));
if (mod(size_filter(1),2) == 0) || (mod(size_filter(2),2) == 0)
    display('The dimensions of your filter have to be odd')
else
    center_filter = find_center(); 
    padded_image = pad_image(image, center_filter);
    
    for idx_i=0:size_image(1)-1
        for idx_j=0:size_image(2)-1
            filtered_image(size_image(1)-idx_i, size_image(2)-idx_j, :) = multiply_add(padded_image, idx_i, idx_j);
        end
    end
        
end
output = filtered_image;

    function centr = find_center()
        centr = zeros(1,2);
        centr(1) = floor(size_filter(1)/2) + 1;
        centr(2) = floor(size_filter(2)/2) + 1;
    end
    function image_padded = pad_image(original_image, filter_center)
       image_padded =  padarray(original_image, [size_filter(1)-filter_center(1), size_filter(2) - filter_center(2)]);
    end

    function pixel = multiply_add(image_padded, row_indx, col_indx)
        if size(image_padded,3)== 3
            pixel = zeros(1,1,3);
            for i=1:size_filter(1)
                for j=1:size_filter(2)
                    pixel(1,1,:) = pixel(1,1,:) + filter(size_filter(1)-i+1,size_filter(2)-j+1)*image_padded(size_image(1)-row_indx + i-1, size_image(2)-col_indx + j-1, :);
                    %pixel(1,1,2) = pixel(1,1,2) + filter(size_filter(1)-i+1,size_filter(2)-j+1)*image_padded(size_image(1)-row_indx + i-1, size_image(2)-col_indx + j-1, 2);
                    %pixel(1,1,3) = pixel(1,1,3) + filter(size_filter(1)-i+1,size_filter(2)-j+1)*image_padded(size_image(1)-row_indx + i-1, size_image(2)-col_indx + j-1, 3);
                end
            end
        else
            pixel = 0;
            for i=1:size_filter(1)
                for j=1:size_filter(2)
                    pixel = pixel + filter(size_filter(1)-i+1,size_filter(2)-j+1)*image_padded(size_image(1)-row_indx + i-1, size_image(2)-col_indx + j-1);
                end
            end
        end
     
    end
    
    
%%%%%%%%%%%%%%%%



end

