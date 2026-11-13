
"""
    convert_ifft(x; dims = [1, 2])

Apply the centred inverse FFT along `dims` and multiply by
`N = prod(size(x)[dims])`. The default transforms the first two dimensions.
The exact convention is `fftshift(ifft(ifftshift(x,dims),dims),dims) * N`.
Together with `convert_fft` and matching dimensions, this forms an inverse
array-transform pair up to floating-point error.
"""
convert_ifft(x;dims=[1,2]) = fftshift(ifft(ifftshift(x,dims),dims),dims)*prod(size(x)[dims])
"""
    convert_fft(x; dims = [1, 2])

Apply the centred FFT along `dims` and divide by
`N = prod(size(x)[dims])`. The default transforms the first two dimensions.
The exact convention is `fftshift(fft(ifftshift(x,dims),dims),dims) / N`.
This is a Cartesian array transform, not the package's physical high-order
encoding operator or its masked normalization.
"""
convert_fft(x;dims=[1,2])  = fftshift( fft(ifftshift(x,dims),dims),dims)/prod(size(x)[dims])
