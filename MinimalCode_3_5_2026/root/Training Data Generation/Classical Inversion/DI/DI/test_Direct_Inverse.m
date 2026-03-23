%% read data
filename = 'D:/zhangjiaying/MRE/code/mrej/Validation_MREJ_pha.raw';
file = fopen(filename,'r');
img0 = fread(file,'float64');
img = reshape(img0,256,256,8);
figure,
for i = 1:8
    subplot(2,4,i)
    img(:,:,i) = img(:,:,i).';
    imshow(img(:,:,i),[])
end
mag = imread('D:/zhangjiaying/MRE/code/mrej/Validation_MREJ_mag.bmp');
%% test DI method
sm = Direct_Inverse(img,1000,[0.2,0.2],100,1000,4);
figure, imshow(sm,[0 8])

