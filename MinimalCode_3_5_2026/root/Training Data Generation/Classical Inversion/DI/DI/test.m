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
%% algebric helmholtz equation
% timesteps = size(img,3);
% if timesteps == 1
%     img_h = img;
% else
%     imgf = fft(img,timesteps,3);
%     img_h = imgf(:,:,2);
% end
% figure,
% for i = 1:8
%     subplot(2,4,i)
%     imshow(imgf(:,:,i),[])
% end
% figure, imshow(img_h,[])

data = img(:,:,1);
freq = 100;
fov = [0.2, 0.2]; % (m)
[nx, ny] = size(data);
pixsize = fov./size(data);
origin = floor(size(data)./2) + 1;
[kx, ky] = meshgrid(1-origin(2):1:ny-origin(2), 1-origin(1):1:nx-origin(1));
kx = 2*pi/ny/pixsize(2)*kx;
ky = 2*pi/nx/pixsize(1)*ky;

data(data==0) = eps;

filter = 1./(1 + (sqrt(kx.^2+ky.^2)/1000).^4);
ft = fftshift(fft2(data));
data_filter = ifft2(ifftshift(filter.*ft));
figure, 
subplot(121), imshow(data,[])
subplot(122), imshow(data_filter,[])


[ux, uy] = gradient(data_filter, pixsize(1), pixsize(2));
[uxx, uyx] = gradient(ux, pixsize(1), pixsize(2));
[uxy, uyy] = gradient(uy, pixsize(1), pixsize(2));

tmp = 2*pi*freq./real(sqrt((uxx+uyy)./(-data)));
figure, imshow(tmp.^2,[0 8])
tmp = tmp(3:end-2, 3:end-2);
cstar = median(tmp(:));
sm = cstar.^2;
figure, imshow(sm,[])

k = real(sqrt((uxx+uyy)./(-data)));
sm = (freq./k).^2;
figure, imshow(sm,[0 0.5])
