# Morlib帮助文档

Morlib基于numpy开发，提供了形态学基本操作的函数和部分其他函数的接口。使用的样例和输入输出可以参考 demo.py

## padding

```py
    def padding(img, kernal_size, border_filled):
```

padding用于在图像周围添加一圈，使得在卷积的前后图像的大小不发生改变。

padding要求的参数如下：
    img {nparray} -- the input binary picture.  
    kernal_size {tuple/list/ndarray} -- the size of the kernel used. Should be 2n+1,n>=0.  
    border_filled {str} --  indicate the way border filled. 'CONSTANT' means filling with 0; 'NEAREST' will filled with nearest value.