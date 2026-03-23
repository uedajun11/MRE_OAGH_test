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
%% extract first harmonic
timesteps = size(img,3);
if timesteps == 1
    img_h = img;
else
    imgf = fft(img,timesteps,3);
    img_h = imgf(:,:,2);
end
figure,
for i = 1:8
    subplot(2,4,i)
    imshow(imgf(:,:,i),[])
end
figure, imshow(img_h,[])
%% Laplacian operator
ac = zeros(0);
nOrder = 3;
nSize = 5;
m = 0;
for j = 0:nOrder
    for i = 0:(nOrder-j)
        m = m+1;
        for y = -(nSize-1)/2:(nSize-1)/2
            for x = -(nSize-1)/2:(nSize-1)/2
                ac(end+1) = x^i * y^j;
            end
        end
    end
end
n = length(ac) / m;
a = zeros(m,n);
for i = 1:m
    for j = 1:n
        a(i,j) = ac((i-1)*n+j);
    end
end
c = a.' * a;
c = a * inv(c);
ans1 = c(1,:);
sg1 = reshape(ans1, nSize, nSize);
ans2 = c(3,:);
sg2 = reshape(ans2, nSize, nSize);
ans3 = c(8,:);
sg3 = reshape(ans3, nSize, nSize);
phi1 = conv2(img_h,sg1,'same');
tmpx1 = conv2(phi,sg2,'same');
tmpy1 = conv2(phi,sg3,'same');
phi = convolve(img_h,sg1);
tmpx = convolve(phi,sg2);
tmpy = convolve(phi,sg3);
[rn, cn] = size(img_h);
lap = zeros(1,rn*cn);
for i = 1:rn
    for j = 1:cn
        lap((i-1)*cn+j) = tmpx(i,j) + tmpy(i,j);
        if abs(lap((i-1)*cn+j)) == 0
            lap((i-1)*cn+j) = 1+1i;
        end
    end
end
%% DI
mf = 100; % mechanical frequency
rho = 1; % density
aa = zeros(1,rn*cn);
bb = zeros(rn,cn);
for i = 1:rn
    for j = 1:cn
        aa((i-1)*cn+j) = img_h(i,j) / lap((i-1)*cn+j);
        aa((i-1)*cn+j) = -(2*pi*mf)^2*rho * aa((i-1)*cn+j);
        bb(i,j) = abs(real(aa((i-1)*cn+j)));
    end
end
fov = [0.2, 0.2];
sm = bb * (fov(1)/cn) * (fov(2)/rn);
figure, imshow(sm, [0 2])
%% Laplacian operator 1.0
img_h = img(:,:,2);
[r,c] = size(img_h);
L = zeros(size(img_h));
for i = 2:r-1
    for j = 2:c-1
        L(i,j) = img_h(i+1,j)+img_h(i-1,j)+img_h(i,j+1)+img_h(i,j-1)-4*img_h(i,j);
    end
end
figure, imshow(L,[])
%% DI 1.0
rho = 1;
mf = 100;
% for i = 1:r
%     for j = 1:c
%         if L(i,j) == 0+0i
%             L(i,j) = 1+0i;
%         end
%     end
% end
a = real(img_h);
b = imag(img_h);
c = real(L);
d = imag(L);
% div = (a.*c+b.*d)./(c.^2-d.^2) + ((b.*c-a.*d)./(c.^2-d.^2))*i;
div = img(:,:,2) ./ L;
sm = -rho * (2*pi*mf)^2 * div;
figure, imshow(sm,[])