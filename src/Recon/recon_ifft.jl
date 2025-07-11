
convert_ifft(x;dims=[1,2]) = fftshift(ifft(ifftshift(x,dims),dims),dims)*prod(size(x)[dims])
convert_fft(x;dims=[1,2])  = fftshift( fft(ifftshift(x,dims),dims),dims)/prod(size(x)[dims])
