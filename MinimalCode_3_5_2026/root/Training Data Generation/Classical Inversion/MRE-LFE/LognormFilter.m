function logfilter = LognormFilter(h,w,bw,rho,fov)

% h: height of lognorm filter
% w: width of lognorm filter
% bw: bandwidth of lognorm filter
% rho: centre frequency of lognorm filter
% fov: field of view

    x_range = linspace(-w/2/fov(1),w/2/fov(1),w);
    y_range = linspace(-h/2/fov(2),h/2/fov(2),h);
    [x,y] = meshgrid(x_range,y_range);
    radius = sqrt(x.^2+y.^2);
    radius(round(h/2)-1:round(h/2)+1,round(w/2)-1:round(w/2)+1) = 1;
    cb = 4/log(2)/bw^2;
    logfilter = exp(-cb*log(radius/rho).^2);
    % logfilter(round(h/2)-1:round(h/2)+1,round(w/2)-1:round(w/2)+1) = 0;

end

