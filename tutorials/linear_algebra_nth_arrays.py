# All of the code copied from https://numpy.org/numpy-tutorials/content/tutorial-svd.html
from scipy import misc

import numpy as np
import matplotlib.pyplot as plt


def linearAlgebraNthArrays() -> None:
    # img is a specific image file in scipy's misc file
    img = misc.face()

    # The type of the img file is a numpy.ndarray
    print(type(img))

    # The shape is a matrix showing the (height, width, RGB color)
    print(img.shape)

    print(img.ndim)

    # Printing out the whole numpy array
    print(img[:,:,0])

    # Getting only the dimensions of img
    print(img[:,:,0].shape)

    # Using matplotlib to plot out img
    print("Original")
    plt.imshow(img)
    plt.show()

    # Here we are making the RGB values as either 0 or 1
    img_array = img/255
    # Finding values
    print(img_array.max(), img_array.min())
    print(img_array.dtype)
    # red_array = img_array[:,:,0]
    # green_array = img_array[:,:,1]
    # blue_array = img_array[:,:,2]
    img_gray = img_array @ [0.2126,0.7152,0.0722]
    print(img_gray.shape)
    print("Gray scale")
    plt.imshow(img_gray,cmap="gray")
    plt.show()

    # Other operations on axis
    U, s, Vt = np.linalg.svd(img_gray)
    print(U, s, Vt)
    print(U.shape, s.shape, Vt.shape)
    # to do s @ Vt
    Sigma = np.zeros((U.shape[1],Vt.shape[0]))
    np.fill_diagonal(Sigma,s)

    # Approximation
    print(np.linalg.norm(img_gray - U @ Sigma @ Vt))
    print(np.allclose(img_gray, U @ Sigma @ Vt))
    print("Aproximation graph")
    plt.plot(s)
    plt.show()

    # Adding in blurriness
    # K is the percentage of blurr applied on image
    k = 5
    approx = U @ Sigma[:,:k] @ Vt[:k,:]
    plt.imshow(approx, cmap="gray")
    plt.show()

    # Applying Colors
    img_array_transposed = np.transpose(img_array, (2,0,1))
    print(img_array_transposed.shape)
    U, s, Vt = np.linalg.svd(img_array_transposed)
    print(U.shape, s.shape, Vt.shape)
    Sigma = np.zeros((3,768,1024))
    for j in range(3):
        np.fill_diagonal(Sigma[j,:,:],s[j,:])
    reconstructed = U @ Sigma @ Vt
    print(reconstructed.shape)
    print(reconstructed.min(),reconstructed.max())
    reconstructed = np.clip(reconstructed, 0, 1)
    plt.imshow(np.transpose(reconstructed, (1,2,0)))
    plt.show()

    # Readding in the approximation
    approx_img = U @ Sigma[...,:k] @ Vt[...,:k,:]
    print(approx_img.shape)
    plt.imshow(np.transpose(approx_img, (1,2,0)))
    plt.show()
