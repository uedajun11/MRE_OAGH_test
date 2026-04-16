function lf = lfe_2d(img,rho_num,dir_num,fov)

% img: input wave image
% rho_num: the number of centre frequencies of lognorm filter
% dir_num: the number of directions of directional filter
% fov: field of view

    imgf = fftshift(fft2(img));
    [h,w] = size(imgf);
    if rho_num <= 8
        bw = 2*sqrt(2);
        rho = 2.^(1:rho_num);
    elseif rho_num <= 16
        bw = 2;
        rho = sqrt(2).^(1:rho_num);
    end
    q1 = zeros(h,w);
    q2 = zeros(h,w);
    for i = 1:rho_num
        q = zeros(h,w);
        logfilter = LognormFilter(h,w,bw,rho(i)/max(fov),fov);
        for d = 1:dir_num
            dirfilter = DirectionalFilter(h,w,dir_num,d-1,fov);
            img_filter = imgf .* logfilter.*  dirfilter;
            q_d = ifft2(ifftshift(img_filter));
            q = q + abs(q_d);
        end
        if i < rho_num
            q1 = q1 + q;
        end
        if i > 1
            q2 = q2 + sqrt(rho(i)*rho(i-1)) * q;
        end
    end
    lf = (q2 ./ q1) ./ max(fov);

end

