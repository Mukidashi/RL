

def conv2d_size_out(size,kernel_size,stride):
    return (size-(kernel_size-1)-1)//stride + 1