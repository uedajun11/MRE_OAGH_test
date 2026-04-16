function imgf_temporal = DirectionFilter_Gaussian(img, TotalDirectionNumber, DirectionIndex, cutin, cutoff, dbBandwith)

if nargin < 6, dbBandwith=3; end
if nargin < 5, cutoff=0.35; end
if nargin < 4, cutin=0.01; end
if nargin < 3, error('Too few input parameters'); end

nOrt = TotalDirectionNumber;
Orz = DirectionIndex;
[n1,n2,s] = size(img);
center_freq = (cutin+cutoff)/2;
singalside_bandwith = (cutoff-cutin)/2;
sigma2 = singalside_bandwith.^2/(dbBandwith/10*log(10));

x = (ones(n1,1)*(1:n2)-(fix(n2/2)+1))/n2;
y = ((1:n1)'*ones(1,n2)-(fix(n1/2)+1))/n1;

radius = sqrt(x.^2+y.^2);
lgdf = exp(-(radius-center_freq).^2/(2*sigma2));

rdtheta = atan2(-y,x);
sintheta = sin(rdtheta);
costheta = cos(rdtheta);

dtheta_on_sigma = sqrt(nOrt);
theta_sigma = 2*pi/nOrt/dtheta_on_sigma;
theta = (2*pi/nOrt)*Orz;

ds = sintheta.*cos(theta)-costheta.*sin(theta);
dc = costheta.*cos(theta)+sintheta.*sin(theta);
dtheta = abs(atan2(ds,dc));

spread = exp((-dtheta.^2) / (2*theta_sigma^2));
lgdf = lgdf.*spread;
imgf = zeros(size(img));

% img_temporalF = fftshift(fft(img,s,3),3);
img_temporalF = img;
for k = 1:s
    imgs = img_temporalF(:,:,k);
    imgfft = fftshift(fft2(imgs));
    imgft = imgfft.*lgdf;
    imgs = ifft2(ifftshift(imgft));
    imgf(:,:,k) = imgs;
end
% imgf_temporal = ifftshift(imgf,3);
imgf_temporal = real(imgf);

end

