"""Week 6 / Day 1: single-channel cross-correlation, stride, and zero padding."""

import torch
from torch.nn import functional as F


def demo_sliding_windows():
    """Example: manually slide a 2x2 kernel over a 3x3 image."""
    image = torch.arange(1, 10, dtype=torch.float32).reshape(3, 3)
    kernel = torch.tensor([[1.0, 0.0], [0.0, -1.0]])
    output = image.new_zeros((2, 2))
    for row in range(2):
        for column in range(2):
            window = image[row:row+2, column:column+2]
            output[row, column] = (window * kernel).sum()
            print("position:", (row, column), "window:", window,
                  "output:", output[row, column].item())
    print("feature map:", output)


def practice_image_shapes():
    """Practice 1: connect a two-dimensional image with the NCHW convention."""
    image = torch.arange(12, dtype=torch.float32).reshape(3, 4)

    # TODO:
    # 1. Add batch and channel axes to get shape (1,1,3,4).
    # 2. Print image/batch shapes and verify batch[0,0] equals image.
    # 3. Slice the 2x2 window starting at row=1, column=1; print its values.
    # 4. Explain why reshape alone does not change HWC into CHW correctly.
    print("Practice 1: complete image shapes and window slicing.")
    # 1. 增加batch、channel维度 → (1,1,3,4)
    img_batch = image.unsqueeze(0).unsqueeze(0)

    # 2
    print("original image shape", image.shape)
    print("batched image shape", img_batch.shape)
    print(torch.all(img_batch[0, 0] == image))  # True

    # 3 截取窗口：起始row=1, col=1，大小2×2
    window = image[1:1 + 2, 1:1 + 2]
    print("window:\n", window)


def output_size(height, width, kernel_height, kernel_width, stride=1, padding=0):
    """Practice 2: return the output height and width for dilation=1."""
    # TODO:
    # 1. Require positive integer dimensions/stride and nonnegative padding.
    # 2. Require padded image dimensions to be at least the kernel dimensions.
    # 3. Compute height/width independently using // stride + 1.

    #1.检查合法性
    assert height>0 and width>0 and kernel_height>0 and kernel_width>0
    assert stride>0 and padding >=0
    # 填充之后图片尺寸
    padded_h = height + 2*padding
    padded_w = width + 2*padding
    # 2.填充后图片不能比kernel更小
    assert padded_h >= kernel_height
    assert padded_w >= kernel_width
    #3 计算输出高宽
    out_h = (padded_h - kernel_height) // stride + 1
    out_w = (padded_w - kernel_width) // stride + 1
    return out_h, out_w


def practice_output_sizes():
    """Practice 2: predict dimensions before running the formula."""
    cases = [
        (5, 5, 3, 3, 1, 0),
        (5, 5, 3, 3, 2, 0),
        (5, 5, 3, 3, 1, 1),
        (6, 7, 3, 2, 2, 1),
    ]
    # TODO: Write each expected shape in a comment, then call output_size.
    print("Practice 2: complete and check the four output-size cases.")
    print("Practice 2: complete and check the four output-size cases.")
    for H, W, Kh, Kw, S, P in cases:
        oh, ow = output_size(H, W, Kh, Kw, S, P)
        print(f"input ({H},{W}), kernel({Kh},{Kw}), stride={S}, pad={P} → output ({oh},{ow})")


def manual_corr2d(image, kernel, stride=1, padding=0):
    """Homework: implement a 2D, one-channel CPU floating-point correlation."""
    # TODO:
    # 1. Check both tensors are 2D and use the same floating dtype/device.
    # 2. Get output dimensions using output_size.
    # 3. Allocate a zero-padded image and copy image into its center.
    # 4. Allocate output; loop over output rows/columns.
    # 5. Start each window at (row*stride, column*stride).
    # 6. Store the sum of window*kernel. Do not flip kernel or apply ReLU.
    # Do not call any existing convolution operator inside this function.
    # 1. Check both tensors are 2D and use the same floating dtype/device.
    assert image.dim() == 2
    assert kernel.dim() == 2
    assert image.dtype == kernel.dtype
    assert image.device == kernel.device

    # 2. Get output dimensions using output_size.
    H, W = image.shape
    Kh, Kw = kernel.shape
    out_h, out_w = output_size(H, W, Kh, Kw, stride, padding)

    # 3. Allocate a zero-padded image and copy image into its center.
    padded_img = torch.zeros((H + 2 * padding, W + 2 * padding), dtype=image.dtype, device=image.device)
    padded_img[padding: padding + H, padding: padding + W] = image

    # 4. Allocate output; loop over output rows/columns.
    output = torch.zeros((out_h, out_w), dtype=image.dtype, device=image.device)

    # 5. Start each window at (row*stride, column*stride).
    for out_r in range(out_h):
        for out_c in range(out_w):
            start_r = out_r * stride
            start_c = out_c * stride
            window = padded_img[start_r: start_r + Kh, start_c: start_c + Kw]
            # 6. Store the sum of window*kernel. Do not flip kernel or apply ReLU.
            output[out_r, out_c] = (window * kernel).sum()
    return output



def verify_manual_implementation():
    """Provided numerical checks: library convolution is only the reference."""
    generator = torch.Generator().manual_seed(7)
    cases = [  # (height, width, kh, kw, stride, padding)
        (3, 3, 2, 2, 1, 0),
        (5, 5, 3, 3, 2, 0),
        (5, 5, 3, 3, 1, 1),
        (6, 7, 3, 2, 2, 1),
        (4, 5, 1, 1, 1, 0),
        (7, 8, 2, 3, 3, 0),
    ]
    for height, width, kh, kw, stride, padding in cases:
        image = torch.randn((height, width), generator=generator)
        kernel = torch.randn((kh, kw), generator=generator)
        actual = manual_corr2d(image, kernel, stride=stride, padding=padding)
        reference = F.conv2d(image[None, None], kernel[None, None],
                             stride=stride, padding=padding)[0, 0]  # PyTorch 内置卷积函数，要求输入必须是 4 维 NCHW：[batch, channel, H, W]
        assert actual.shape == reference.shape
        assert torch.allclose(actual, reference, atol=1e-5, rtol=1e-5)
        print("passed:", (height, width), (kh, kw), "stride/padding:",
              (stride, padding), "output shape:", actual.shape)


def homework_convolution():
    """Homework: run the manual implementation, explain its indexing, and verify."""
    image = torch.arange(1, 10, dtype=torch.float32).reshape(3, 3)
    kernel = torch.tensor([[1.0, 0.0], [0.0, -1.0]])

    # TODO:
    # 1. Complete output_size and manual_corr2d.
    # 2. Print output for stride=1,padding=0; compare with the example.
    # 3. Try stride=2,padding=1; predict the shape before running.
    # 4. Call verify_manual_implementation; all six checks should pass.
    # 5. Explain cross-correlation vs mathematical convolution in comments.
    #
    # A single output value comes from:
    # Stride affects window positions because:
    # Padding affects output size because:
    # Why the kernel is not flipped here:
    print("Homework: complete manual correlation and run the supplied checks.")
    # 1. Complete output_size and manual_corr2d.
    # 2. Print output for stride=1,padding=0; compare with the example.
    out1 = manual_corr2d(image, kernel, stride=1, padding=0)
    print("stride=1,padding=0 output:\n", out1)
    # 3. Try stride=2,padding=1; predict the shape before running.
    out2 = manual_corr2d(image, kernel, stride=2, padding=1)
    print("stride=2,padding=1 output:\n", out2)

    # 4. Call verify_manual_implementation; all six checks should pass.
    verify_manual_implementation()

    # 5. Explain cross-correlation vs mathematical convolution in comments.
    # A single output value comes from:
    # 在填充后的图片上截取一块窗口，窗口和kernel逐元素相乘，所有元素求和。
    # Stride affects window positions because:
    # stride是窗口每次滑动的步长；stride越大，窗口起始位置越少，输出特征图尺寸越小。
    # Padding affects output size because:
    # padding在原图四周补0，扩大图像尺寸，让更多窗口可以放下，输出尺寸变大。
    # Why the kernel is not flipped here:
    # 深度学习CNN使用的是互相关cross-correlation，不需要翻转kernel；数学上严格卷积需要翻转kernel。
    # 网络可以学习等效kernel，所以深度学习框架默认直接做互相关。


if __name__ == "__main__":
    print("--- Example: sliding windows ---")
    demo_sliding_windows()
    print("\n--- Practice 1: image shapes ---")
    practice_image_shapes()
    print("\n--- Practice 2: output dimensions ---")
    practice_output_sizes()
    print("\n--- Homework: manual correlation ---")
    homework_convolution()
