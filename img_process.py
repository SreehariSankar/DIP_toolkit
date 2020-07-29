import numpy as np
import matplotlib.pyplot as plt
import ctypes
import scipy.signal as sg
from sklearn.cluster import KMeans


def split(img):
    """
    To split the image into 4 pieces and get it back such that no complex operation is needed to manipulate pixels
    """
    a0 = np.flip(img[:img.shape[0] // 2, :img.shape[1] // 2])
    a2 = np.flip(img[img.shape[0] // 2:, :img.shape[1] // 2], axis=1)
    a3 = img[img.shape[0] // 2:, img.shape[1] // 2:]
    a1 = np.flip(img[:img.shape[0] // 2, img.shape[1] // 2:], axis=0)
    return a0,a1,a2,a3

def stitch_back(a0,a1,a2,a3,verbose=True):
    """
    Opposite of split(img)
    """
    img1 = np.hstack((np.flip(a0), np.flip(a1, axis=0)))
    img2 = np.hstack((np.flip(a2, axis=1), a3))
    img3 = np.vstack((img1, img2))
    if verbose:
        plt.matshow(img3, cmap='gray')
        plt.show()
    return img3

def readraw_color(path="Images/library_color.raw", verbose=False):
    data_array = np.fromfile(path,dtype=ctypes.c_ubyte)
    data=np.asarray(data_array)
    a=len(data)
    fnl=[]
    N=np.int(np.sqrt(a/3))
    # print(N,":N")
    for i in range(3):
        n=np.arange(i,a,3)
        d=data[n]
        # print(d.shape," D shape")
        d=np.reshape(d,(N,N))
        fnl.append(d)
    fnl=np.asarray(fnl)
    fnl=np.stack((fnl[0],fnl[1],fnl[2]),axis=2)
    # 0,1,2
    if verbose:
        plt.imshow(fnl)
        plt.show()
    return fnl

def readraw(path="Images/cwru_logo_gray.raw", verbose=False):
    data_array = np.fromfile(path,dtype=ctypes.c_ubyte)
    data=np.asarray(data_array)
    a=len(data)
    N=np.int(np.sqrt(a))
    # print(N,":N")
    d=data
    # print(d.shape," D shape")
    d=np.reshape(d,(N,N))
    if verbose:
        plt.imshow(d,cmap='gray')
        plt.suptitle("Original")
        plt.show()
    return d

def readraw_neg(path="Images/library_color.raw", verbose=False):
    data_array = np.fromfile(path, dtype=ctypes.c_ubyte)
    data = np.asarray(data_array)
    a = len(data)
    fnl = []
    N = np.int(np.sqrt(a / 3))
    for i in range(3):
        n = np.arange(i, a, 3)
        d = data[n]
        d = np.reshape(d, (N, N))
        for i in range(d.shape[0]):
            for j in range(d.shape[1]):
                d[i,j]=255-d[i,j]
        fnl.append(d)

    fnl = np.asarray(fnl)
    fnl = np.stack((fnl[0], fnl[1], fnl[2]), axis=2)
    # 0,1,2
    if verbose:
        plt.imshow(fnl)
        plt.show()
    return fnl

def filter_grey(image,flt_):
    """
    convolution for (N,M,1) image
    """
    im = image
    new_col = np.empty(im[1,:].shape)
    for j in range(im.shape[0]):
        new_row = []
        for k in range(im.shape[1]):
            x, y = flt_.shape
            pixel = 0
            for p in range(x):
                for l in range(y):
                    pixel += (im[abs(j - p)][abs(k - l)] * flt_[p][l])
            new_row.append(pixel)
        new_col = np.vstack([new_col, new_row])
    new_col = new_col[1:]
    return new_col

def thresh(img,k):
    img[img>=k]=255
    img[img<k]=0
    return img

def neighbors(img,i,j):
    if i == (img.shape[0] - 1):
        if j == (img.shape[1] - 1):
            a0 = img[i, j]
            a1 = img[i, j - 1]
            a2 = img[i, j - 1]
            a3 = img[i, j]
            a4 = img[i, j]
            a5 = img[i, j]
            a6 = img[i - 1, j]
            a7 = img[i - 1, j]
            a8 = img[i - 1, j - 1]
        else:
            a0 = img[i, j]
            a1 = img[i, j - 1]
            a2 = img[i, j - 1]
            a3 = img[i, j]
            a4 = img[i, j + 1]
            a5 = img[i, j + 1]
            a6 = img[i - 1, j + 1]
            a7 = img[i - 1, j]
            a8 = img[i - 1, j - 1]
    elif j == (img.shape[1] - 1):
        a0 = img[i, j]
        a1 = img[i, j - 1]
        a2 = img[i + 1, j - 1]
        a3 = img[i + 1, j]
        a4 = img[i + 1, j]
        a5 = img[i, j]
        a6 = img[i - 1, j]
        a7 = img[i - 1, j]
        a8 = img[i - 1, j - 1]
    else:
        a0 = img[i, j]
        a1 = img[i, j - 1]
        a2 = img[i + 1, j - 1]
        a3 = img[i + 1, j]
        a4 = img[i + 1, j + 1]
        a5 = img[i, j + 1]
        a6 = img[i - 1, j + 1]
        a7 = img[i - 1, j]
        a8 = img[i - 1, j - 1]

    mat = np.asarray([[a8, a7, a6], [a1, a0, a5], [a2, a3, a4]])
    return mat

def hist_transfer(img, bits=256):
    '''
    Histogram transfeer for a given number of output bits
    '''
    x = np.zeros(bits)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            x[int(img[i, j])] += 1
    x=x/(img.shape[0]*img.shape[1])
    img_cdf=x.cumsum()
    bin_centres=np.arange(bits)
    out = np.interp(img.flat, bin_centres, img_cdf)
    o=out.reshape(img.shape)
    tr = np.uint8(bits * img_cdf)
    plt.plot(tr)
    plt.show()
    return o

def plot_histo(img, ret=False, silence=False):
    '''
    Same as matplotlib.pyplot.hist
    '''
    x=np.zeros(256)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            x[int(img[i,j])]+=1
    if silence==False:
        plt.plot(x)
        plt.show()
    if ret:
        return x


def norm(ar):
    return 255.*np.absolute(ar)/np.max(ar)


def translation(img,tx,ty,time=1):
    """
    Translates image, 0 is added for lost indices
    :param img: Input image
    :param tx: x units of translation
    :param ty: y units of translation
    :param time: time step of translation
    :param iter: iterative nature of time step
    :return: translated image (with display)
    """
    img1=np.zeros(img.shape)
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            if ((i+int(tx*time))>=img.shape[0]) or ((j+int(ty*time))>=img.shape[1]) or ((i+int(tx*time)) < 0)or ((j+int(ty*time))< 0):
                img1[i, j] = 0
            else:
                img1[i, j] = img[i + int(tx*time)][j + int(ty*time)]

    # plt.matshow(img1,cmap='gray')
    # plt.show()
    return img1

def rotation(img, theta, time=1):
    # THETA DEGREES PER SECOND
    theta=-1*time*theta*np.pi/180
    cos=np.cos(theta)
    sin=np.sin(theta)
    xc=img.shape[0]//2
    yc=img.shape[1]//2
    img1=np.zeros(img.shape)
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            xj=int(cos*i - sin*j + xc + yc*sin - xc*cos)
            yk=int(sin*i + cos*j + yc - yc*sin - xc*cos)
            if (xj>=img.shape[0]) or (yk>=img.shape[1]) or (xj < 0)or (yk< 0):
                img1[i, j] = 0
            else:
                img1[i, j] = img[xj][yk]
    # plt.matshow(img1,cmap='gray')
    # plt.show()
    return img1

def scaling(img, s, time=1, verbose=True):
    a0=np.flip(img[:img.shape[0]//2,:img.shape[1]//2])
    a2=np.flip(img[img.shape[0]//2:,:img.shape[1]//2],axis=1)
    a3=img[img.shape[0]//2:,img.shape[1]//2:]
    a1=np.flip(img[:img.shape[0]//2,img.shape[1]//2:],axis=0)
    a0=scaling_safe(a0,s,time,False)
    a1=scaling_safe(a1,s,time,False)
    a2=scaling_safe(a2,s,time,False)
    a3=scaling_safe(a3,s,time,False)
    img1=np.hstack((np.flip(a0),np.flip(a1,axis=0)))
    img2=np.hstack((np.flip(a2,axis=1),a3))
    img3=np.vstack((img1,img2))

    return img3

def warp_rect_to_circle(img, verbose=True):
    a0,a1,a2,a3=split(img)
    na0=np.zeros(a0.shape)
    na1=np.zeros(a1.shape)
    na2=np.zeros(a2.shape)
    na3=np.zeros(a3.shape)
    for i in range(1,a0.shape[0]):
        for j in range(1,a0.shape[1]):
            if i ** 2 >= j ** 2:
                u = int(i ** 2 / (np.sqrt(i ** 2 + j ** 2)))
                v = int(i * j / (np.sqrt(i ** 2 + j ** 2)))
            else:
                v = int(j ** 2 / (np.sqrt(i ** 2 + j ** 2)))
                u = int(i * j / (np.sqrt(i ** 2 + j ** 2)))
            na0[u, v]=a0[i, j]
    for i in range(1, a1.shape[0]):
        for j in range(1, a1.shape[1]):
            if i ** 2 >= j ** 2:
                u = int(i ** 2 / (np.sqrt(i ** 2 + j ** 2)))
                v = int(i * j / (np.sqrt(i ** 2 + j ** 2)))
            else:
                v = int(j ** 2 / (np.sqrt(i ** 2 + j ** 2)))
                u = int(i * j / (np.sqrt(i ** 2 + j ** 2)))
            na1[u, v]=a1[i, j]
    for i in range(1, a2.shape[0]):
        for j in range(1, a2.shape[1]):
            if i ** 2 >= j ** 2:
                u = int(i ** 2 / (np.sqrt(i ** 2 + j ** 2)))
                v = int(i * j / (np.sqrt(i ** 2 + j ** 2)))
            else:
                v = int(j ** 2 / (np.sqrt(i ** 2 + j ** 2)))
                u = int(i * j / (np.sqrt(i ** 2 + j ** 2)))
            na2[u, v]=a2[i, j]
    for i in range(1, a3.shape[0]):
        for j in range(1, a3.shape[1]):
            if i ** 2 >= j ** 2:
                u = int(i ** 2 / (np.sqrt(i ** 2 + j ** 2)))
                v = int(i * j / (np.sqrt(i ** 2 + j ** 2)))
            else:
                v = int(j ** 2 / (np.sqrt(i ** 2 + j ** 2)))
                u = int(i * j / (np.sqrt(i ** 2 + j ** 2)))
            na3[u, v]=a3[i, j]

    img3=stitch_back(na0,na1,na2,na3,False)
    if verbose:
        plt.matshow(img3, cmap='gray')
        plt.suptitle('Rectange to circle')
        plt.show()
    return img3

def warp_circle_to_rect(img,verbose=True):
    a0, a1, a2, a3 = split(img)
    na0 = np.zeros(a0.shape)
    na1 = np.zeros(a1.shape)
    na2 = np.zeros(a2.shape)
    na3 = np.zeros(a3.shape)
    for i in range(1, a0.shape[0]):
        for j in range(1, a0.shape[1]):
            if i ** 2 >= j ** 2:
                u = int((np.sqrt(i ** 2 + j ** 2)))
                v = int((j/i) * (np.sqrt(i ** 2 + j ** 2)))

            else:
                v = int((np.sqrt(i ** 2 + j ** 2)))
                u = int((i / j) * (np.sqrt(i ** 2 + j ** 2)))

            if v > 255:
                v = 255
            if u > 255:
                u = 255

            na0[u, v] = a0[i, j]
    for i in range(1, a1.shape[0]):
        for j in range(1, a1.shape[1]):
            if i ** 2 >= j ** 2:
                u = int((np.sqrt(i ** 2 + j ** 2)))
                v = int((j / i) * (np.sqrt(i ** 2 + j ** 2)))
            else:
                v = int((np.sqrt(i ** 2 + j ** 2)))
                u = int((i / j) * (np.sqrt(i ** 2 + j ** 2)))
            if v > 255:
                v = 255
            if u > 255:
                u = 255

            na1[u, v] = a1[i, j]
    for i in range(1, a2.shape[0]):
        for j in range(1, a2.shape[1]):
            if i ** 2 >= j ** 2:
                u = int((np.sqrt(i ** 2 + j ** 2)))
                v = int((j / i) * (np.sqrt(i ** 2 + j ** 2)))
            else:
                v = int((np.sqrt(i ** 2 + j ** 2)))
                u = int((i / j) * (np.sqrt(i ** 2 + j ** 2)))
            if v > 255:
                v = 255
            if u > 255:
                u = 255

            na2[u, v] = a2[i, j]
    for i in range(1, a3.shape[0]):
        for j in range(1, a3.shape[1]):
            if i ** 2 >= j ** 2:
                u = int((np.sqrt(i ** 2 + j ** 2)))
                v = int((j / i) * (np.sqrt(i ** 2 + j ** 2)))
            else:
                v = int((np.sqrt(i ** 2 + j ** 2)))
                u = int((i / j) * (np.sqrt(i ** 2 + j ** 2)))
            if v > 255:
                v = 255
            if u > 255:
                u = 255
            na3[u, v] = a3[i, j]

    img3 = stitch_back(na0, na1, na2, na3, False)
    if verbose:
        plt.matshow(img3, cmap='gray')
        plt.suptitle('Circle to rectangle')
        plt.show()
    return img3

def laws(img):
    # LAWS FILTER FOR TEXTURE
    img1=img
    img2 = np.copy(img1)
    (rows, cols) = img1.shape
    cm = np.zeros((rows, cols, 16), np.float64)
    fv = np.array([[1, 4, 6, 4, 1],
                               [-1, -2, 0, 2, 1],
                               [-1, 0, 2, 0, 1],
                               [1, -4, 6, -4, 1]])
    filters = []
    for i in range(4):
        for j in range(4):
            filters.append(np.matmul(fv[i][:].reshape(5, 1), fv[j][:].reshape(1, 5)))
    sk = (1 / 25) * np.ones((5, 5))
    smoothed = sg.convolve(img2, sk, "same")
    processed = np.abs(img2 - smoothed)

    for i in range(len(filters)):
        cm[:, :, i] = sg.convolve(processed, filters[i], 'same')

    tm = []
    tm.append(norm((cm[:, :, 1] + cm[:, :, 4]) // 2))
    tm.append(norm((cm[:, :, 3] + cm[:, :, 12]) // 2))
    tm.append(norm(cm[:, :, 10]))
    tm.append(norm(cm[:, :, 15]))
    tm.append(norm((cm[:, :, 2] + cm[:, :, 8]) // 2))
    tm.append(norm(cm[:, :, 5]))
    tm.append(norm((cm[:, :, 7] + cm[:, :, 13]) // 2))
    tm.append(norm((cm[:, :, 11] + cm[:, :, 14]) // 2))


    plt.figure('Images (Preprocess)')
    plt.subplot(221)
    plt.imshow(img,cmap='gray')
    plt.subplot(222)
    plt.imshow(img1, 'gray')
    plt.subplot(223)
    plt.imshow(smoothed, 'gray')
    plt.subplot(224)
    plt.imshow(processed, 'gray')

    plt.figure('Texture Maps. Set 1')
    plt.subplot(221)
    plt.imshow(tm[0], 'gray')
    plt.subplot(222)
    plt.imshow(tm[1], 'gray')
    plt.subplot(223)
    plt.imshow(tm[2], 'gray')
    plt.subplot(224)
    plt.imshow(tm[3], 'gray')

    plt.figure('Texture Maps. Set 2')
    plt.subplot(221)
    plt.imshow(tm[4], 'gray')
    plt.subplot(222)
    plt.imshow(tm[5], 'gray')
    plt.subplot(223)
    plt.imshow(tm[6], 'gray')
    plt.subplot(224)
    plt.imshow(tm[7], 'gray')
    plt.show()


def texture_clustering(raw_image,n_k=5):
    # TEXTURE CLUSTERING WITH KMEANS

    filters = [np.array([1, 4, 6, 4, 1]).reshape(1, 5),
               np.array([-1, -2, 0, 2, 1]).reshape(1, 5),
               np.array([-1, 0, 2, 0, -1]).reshape(1, 5),
               np.array([-1, 2, 0, -2, 1]).reshape(1, 5),
               np.array([1, -4, 6, -4, 1]).reshape(1, 5)]
    kernels = []
    vectors = []
    for i in range(5):
        for j in range(5):
            kernels.append(np.matmul(np.transpose(filters[i]), filters[j]))
    img = raw_image - np.mean(raw_image)
    img_extended = np.pad(img, 2, 'reflect')
    filtered_img = []
    for index, kernel in enumerate(kernels):
        filtered_img1 = np.zeros(img.shape)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                filtered_img1[i][j] = convolution(i, j, kernel, img_extended)
        filtered_img.append(filtered_img1)
    filtered = []
    for image in filtered_img:
        rimg = np.pad(image, n_k, 'reflect')
        filtered.append(rimg)

    for a in range(img.shape[0]):
        for b in range(img.shape[1]):
            features = []
            for x in filtered:
                features.append(np.sum(abs(x[a:a + 13, b:b + 13]) - np.mean(x[a:a + 13, b:b + 13])) / 169)
            vectors.append(np.array(features))

    vectors = np.array(vectors)
    vectors = vectors[:, 1:]
    vectors = (vectors - np.mean(vectors, axis=0)) / np.std(vectors, axis=0)
    kmeans = KMeans(n_clusters=n_k).fit(vectors)
    final = kmeans.labels_.reshape(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if final[i][j] == 1:
                final[i][j] = 0
            elif final[i][j] == 2:
                final[i][j] = 63
            elif final[i][j] == 3:
                final[i][j] = 127
            elif final[i][j] == 4:
                final[i][j] = 191
            elif final[i][j] == 5:
                final[i][j] = 255
    final = np.uint8(final)
    plt.matshow(final, cmap='gray')
    plt.show()

def floyd_steinberg(im):

    pixel = np.copy(im)
    w, h = im.shape
    for x in range(w):
        for y in range(h):
            oldpixel = pixel[x][y]
            newpixel = find_closest_palette_color(oldpixel)
            pixel[x][y] = newpixel
            quant_error = oldpixel - newpixel
            if x + 1 < w-1:
                pixel[x + 1][y] = pixel[x + 1][y] + quant_error * 7 / 16
            if x > 0 and y < h-1:
                pixel[x - 1][y + 1] = pixel[x - 1][y + 1] + quant_error * 3 / 16
            if y < h-1:
                pixel[x][y + 1] = pixel[x][y + 1] + quant_error * 5 / 16
            if x < w-1 and y < h-1:
                pixel[x + 1][y + 1] = pixel[x + 1][y + 1] + quant_error * 1 / 16
    plt.imshow(pixel, cmap='gray')
    plt.suptitle("Floyd-Steinberg")
    plt.show()
    return pixel

def test_dith(img,t4):
    # TEST OUT A DITHERING MATRIX

    img4 = copy.copy(img)
    for i in range(img4.shape[0]):
        for j in range(img4.shape[1]):
            if img4[i, j] > t4[i % t4.shape[0]][j % t4.shape[1]]:
                img4[i, j] = 1
            else:
                img4[i, j] = 0
    plt.imshow(img, cmap='gray')
    plt.suptitle("Original")
    plt.show()
    plt.imshow(img4, cmap='gray')
    plt.suptitle("Dithering matrix, i=4")
    plt.show()
    return img4


def find_closest_palette_color(oldpixel):
    return int(round(oldpixel / 255) * 255)

def fixed_dithering(img,t=127):

    # DOES NOT RETURN ANYTHING.
    plt.imshow(thresh(img,t),cmap='gray')
    plt.suptitle("Fixed threshold dithering")
    plt.show()

def rand_dithering_uniform(img):
    random.seed()
    t=random.randint(0,255)
    print(t)
    img1=copy.deepcopy(img)
    img1[img1>=t]=255
    img1[img1<t]=0
    plt.imshow(img1,cmap='gray')
    plt.suptitle("Uniform distribution")
    plt.show()
    return img1

import skimage.filters as skf

sobel_r=(1/4)*np.asarray([[1,0,-1],[2,0,-2],[1,0,-1]])
sobel_c=(1/4)*np.asarray([[-1,-2,-1],[0,0,0],[1,2,1]])
laplacian=(1/8)*np.asarray([[-2,1,-2],[1,4,1],[-2,1,-2]])
def fo(loc,adv=0):

    # FIRST ORDER EDGE DETECTION
    img=readraw(loc,True)
    if adv == 1:
        img = denoise(img)

    img_r=filter_grey(img,sobel_r)
    img_r=thresh(img_r,int(max(np.ndarray.flatten(img_r))*0.2))
    plt.imshow(img_r,cmap='gray')
    plt.suptitle("With Sobel Operator, row")
    plt.show()
    img_c=filter_grey(img,sobel_c)
    img_c=thresh(img_c,int(max(np.ndarray.flatten(img_c))*0.2))
    plt.imshow(img_c,cmap='gray')
    plt.suptitle("With Sobel Operator, column")
    plt.show()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i][j]=np.sqrt(((img_r[i][j]**2) + (img_c[i][j]**2)))
    plt.imshow(img,cmap='gray')
    plt.suptitle("Final edge detection, First order")
    plt.show()


def so(loc,adv=0):
    """
    SECOND ORDER LoG of an image
    """

    img = readraw(loc, True)
    if adv == 1:
        img = denoise(img)

    img = skf.gaussian(img)
    img = filter_grey(img,laplacian)
    plt.imshow(img, cmap='gray')
    plt.suptitle("Second order, LoG")
    plt.show()

def denoise(img):
    M3 = np.asarray([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) * (1 / 16)
    img=filter_grey(img,M3)
    return img
