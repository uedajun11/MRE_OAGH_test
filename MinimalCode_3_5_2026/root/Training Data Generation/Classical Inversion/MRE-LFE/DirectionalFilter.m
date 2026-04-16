function dirfilter = DirectionalFilter(h,w,dir_num,ord,fov)

% h: height of directional filter
% w: width of directional filter
% dir_num: the whole number of direction
% ord: order of directions (from 0 to 2pi)
% fov: field of view

    x_range = linspace(-w/2/fov(1),w/2/fov(1),w);
    y_range = linspace(-h/2/fov(2),h/2/fov(2),h);
    [x,y] = meshgrid(x_range,y_range);
    T = atan2(-y,x);
    theta = 2*pi*ord/dir_num;
    theta_sigma = 2*pi/dir_num;
    ds = sin(T).*cos(theta) - cos(T).*sin(theta);
    dc = cos(T).*cos(theta) + sin(T).*sin(theta);
    dtheta = abs(atan2(ds,dc));
    dirfilter = exp((-dtheta.^2) / (2*theta_sigma.^2));

end

