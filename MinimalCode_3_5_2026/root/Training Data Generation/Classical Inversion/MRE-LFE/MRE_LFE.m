function sm = MRE_LFE(img,rho_num,dir_num,fov,med_ord,mf,opt)

% img: input wave images
% rho_num: the number of centre frequencies of lognorm filter
% dir_num: the number of directions of directional filter
% fov: field of view
% med_ord: the order of median filter
% mf: mechanical frequency (driver frequency) 
% opt: options of image types

    [h,w,s] = size(img);
    img0 = zeros(h,w,s);
    for i = 1:s
        max_img = max(max(img(:,:,i)));
        min_img = min(min(img(:,:,i)));
        img0(:,:,i) = (img(:,:,i) - min_img) ./ (max_img - min_img);
    end
    img0 = complex(img0);
    if opt == "comp"
        img0 = Harmonic(img0,2);
        local_f = lfe_2d(img0,rho_num,dir_num,fov);
    elseif opt == "noncomp"
        img0 = Harmonic(img0,2);
        imgr = real(img0);
        imgi = imag(img0);
        lfr = lfe_2d(imgr,rho_num,dir_num,fov);
        lfi = lfe_2d(imgi,rho_num,dir_num,fov);
        local_f = (lfr + lfi) / 2;
    elseif opt == "temp"
        local_f = zeros(h,w);
        for i = 1:s
            lf = lfe_2d(img0(:,:,i),rho_num,dir_num,fov);
            local_f = local_f + lf ./ s;
        end
    end
    for i = 1:h
        for j = 1:w
            if local_f(i,j) == 0
                local_f(i,j) = local_f + 1;
            end
        end
    end
    lf_med = medfilt2(real(local_f),[med_ord,med_ord],'symmetric');
    sm = (double(mf) ./ (lf_med)).^2;
        
end

