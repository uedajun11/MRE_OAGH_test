function output = convolve(input,kernel)

    [ri,ci] = size(input);
    [rk,ck] = size(kernel);
    if ((ri+rk)-2 && (ri+rk)-1) ~= 0
        rr = 2^nextpow2((ri+rk)-1);
    else
        rr = (ri+rk)-1;
    end
    if ((ci+ck)-2 && (ci+ck)-1) ~= 0
        cc = 2^nextpow2((ci+ck)-1);
    else
        cc = (ci+ck)-1;
    end
    A = zeros(rr,cc);
    B = zeros(rr,cc);
    A(1:ri,1:ci) = input(:,:);
    A(ri+1:end,ci+1:end) = 0+0i;
    B(1:rk,1:ck) = kernel(:,:)+0i;
    B(rk+1:end,ck+1:end) = 0+0i;
    fft_A = fft2(A);
    fft_B = fft2(B);
    fft_C = fft_B .* fft_A;
    C = ifft2(fft_C);
    c1 = zeros((ri+rk)-1,(ci+ck)-1);
    c1(:,:) = C(1:(ri+rk)-1,1:(ci+ck)-1);
    c2 = rr * c1(1+floor(rk/2):ri+floor(rk/2),1+floor(1:ck/2):ci+floor(1:ck/2));
    R = round(real(c2));
    I = round(imag(c2));
    output = R + I*1i;
end

function output = nextpow2(n)
    output = 1;
    for i = 1:floor(log2(n))
        output = output+1;
    end
end
