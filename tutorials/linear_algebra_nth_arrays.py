from numpy import linalg
from scipy import misc

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
