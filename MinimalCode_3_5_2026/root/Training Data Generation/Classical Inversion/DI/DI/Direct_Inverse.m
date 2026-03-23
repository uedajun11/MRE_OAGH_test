function sm = Direct_Inverse(wave, k_filter, fov, mfreq, rho, ws)
% wave: the input wave images
% k_filter: Butterworth filter (1/m)
% fov: field of view (m)
% mfreq: mechanical frequency (Hz)

% extract first harmoni

timesteps = size(wave,3);
if timesteps == 1
    wave_H = wave;
else 
    wave_F = fft(wave,timesteps,3);
    wave_H = wave_F(:,:,2);
end

[nx,ny] = size(wave_H);
pixsize = fov ./ [nx,ny];

origin = floor([nx,ny]./2) + 1;
[kx, ky] = meshgrid(1-origin(2):1:ny-origin(2), 1-origin(1):1:nx-origin(1));
kx = 2*pi/ny/pixsize(2)*kx;
ky = 2*pi/nx/pixsize(1)*ky;
filter = 1./(1+(sqrt(kx.^2+ky.^2)/k_filter).^4);

wave_H(wave_H==0) = eps;
ft = fftshift(fft2(wave_H));
data_filter = ifft2(ifftshift(filter.*ft));

[ux, uy] = gradient(data_filter, pixsize(1), pixsize(2));
[uxx, ~] = gradient(ux, pixsize(1), pixsize(2));
[~, uyy] = gradient(uy, pixsize(1), pixsize(2));

k = real(sqrt((uxx+uyy)./(-wave_H)));
sm = rho * ((2*pi*double(mfreq)) ./ k).^2 ./ 1000 ; % shear modulus(kPa)

sm = medfilt2(sm, [ws,ws], 'zeros');

end

