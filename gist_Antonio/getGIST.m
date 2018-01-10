% EXAMPLE 2
% Load image (this image is not square)
img2 = imread('input.png');

% Parameters:
clear param 
%param.imageSize. If we do not specify the image size, the function LMgist
%   will use the current image size. If we specify a size, the function will
%   resize and crop the input to match the specified size. This is better when
%   trying to compute image similarities.
param.imageSize = [512 512]; % it works also with non-square images
param.orientationsPerScale = [6 6 6 6 6];
param.numberBlocks = 4;
param.fc_prefilt = 4;

% Computing gist requires 1) prefilter image, 2) filter image and collect
% output energies
[gist, param] = LMgist(img2, '', param);

% Visualization
% figure
% subplot(121)
% imshow(img2)
% title('Input image')
% subplot(122)
% showGist(gist, param)
% title('Descriptor')



