import os
import cv2
import numpy as np
import glob

#app_win_size_x = 870
#app_win_size_y = 500
#γ_range_upper_bound = 5

output_dir_path = output_sobel_path = output_log_path = ""

@staticmethod
def root_path(): #當前 working dir 之 root path
    return os.getcwd()
    #return "/workspaces/mvl/ImageProcessing"

def set_output_path():
    global output_dir_path, output_sobel_path, output_log_path
    output_dir_path = os.path.join(root_path(), "hw4_output")
    output_sobel_path = os.path.join(output_dir_path, "sobel")
    output_log_path = os.path.join(output_dir_path, "log")
    
def get_output_path():
    global output_dir_path, output_sobel_path, output_log_path
    return [output_dir_path, output_sobel_path, output_log_path]

def mkdir(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
        print("mkdir "+dir_path)    
    else:
        print(dir_path+" already exist, no need to mkdir.")

def get_image_path(path): #root_path/HW2_test_image
    return glob.glob(os.path.join(path, "*.bmp"))+glob.glob(os.path.join(path, "*.tif"))+glob.glob(os.path.join(path, "*.jpg"))

def show_img_fullscreen(img_name, showimg ,type):
    cv2.namedWindow(img_name, type)
    cv2.setWindowProperty(img_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.resizeWindow(img_name, app_win_size_x,app_win_size_y)
    #cv2.moveWindow(img_name, app_pos_x,app_pos_y)
    cv2.imshow(img_name, showimg)

def read_and_operate_image(image_path):
    image =cv2.imread(image_path)
    #show_img_fullscreen("Current Image: "+image_path, image, cv2.WINDOW_KEEPRATIO)
    image_gray =cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    #show_img_fullscreen("Current Image(grayscale): "+image_path, image_gray, cv2.WINDOW_KEEPRATIO)

    image_RGB = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
    return image, image_gray, image_RGB

#algorithm implementation
#convolution
def convolution(image, filter):
    image_height = image.shape[0]
    image_width = image.shape[1]

    H = (filter.shape[0] -1)//2
    W = (filter.shape[1] -1)//2

    result = np.zeros((image_height, image_width))
    # iterate over all the pixel of image X
    for i in np.arange(H, image_height-H):
        for j in np.arange(W, image_width-W):
            sum = 0
            # iterate over the filter
            for k in np.arange(-H, H+1):
                for l in np.arange(-W, W+1):
                    # get the corresponding value from image and filter
                    a = image[i+k, j+l]
                    w = filter[H+k, W+l]
                    sum += (w * a)
            result[i, j] = sum
    # return convolution
    return result

#sobel operator
def sobel_operator(image):
    #sobel_filter
    Sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Sy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    
    Gx = convolution(image, Sx)
    Gy = convolution(image, Sy)
    result = np.sqrt(np.power(Gx, 2) + np.power(Gy, 2))
    result = (result/np.max(result)) * 255
    return result

#laplacian of gaussian
def laplacian_of_gaussian(image):
    #default輸入為gray image
    #Gaussian降噪
    blur = cv2.GaussianBlur(image, (3,3), 0)
    #laplavcain做edge detection
    dst = cv2.Laplacian(blur, cv2.CV_16S, ksize = 3)
    result = cv2.convertScaleAbs(dst)
    return result,dst

# main
image_dir = os.path.join(root_path(), "HW4_test_image")
print(image_dir)
images = get_image_path(image_dir)  #取得圖片路徑
print(images)

set_output_path()
for output_path in get_output_path():
    mkdir(output_path)

dir, sobel, log = get_output_path()

for image in images:
    file = image.replace(".bmp", "").replace(".tif", "").replace(".jpg", "")
    img, img_gray, img_RGB= read_and_operate_image(image)
    
    img_sobel = sobel_operator(img_gray)
    img_log1,img_log2 = laplacian_of_gaussian(img_gray)
    cv2.imwrite(file.replace(image_dir, sobel)+"_sobel.bmp", img_sobel)
    cv2.imwrite(file.replace(image_dir, log)+"_log1.bmp", img_log1)
    cv2.imwrite(file.replace(image_dir, log)+"_log2.bmp", img_log2)