function img_h = Harmonic(img,order)

% img: input wave images
% order: the order of harmonic extracted

    timesteps = size(img,3);
    if timesteps == 1
        img_h = img;
    else
        imgf = fft(img,timesteps,3);
        img_h = imgf(:,:,order);
    end
        
end

