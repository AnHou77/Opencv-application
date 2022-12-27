# M11015Q02 柯元豪
import cv2
from glob import glob
import os
import copy
from matplotlib import pyplot as plt

def get_contours(image):
    sobelX, sobelY = cv2.Sobel(image,cv2.CV_16S,1,0), cv2.Sobel(image,cv2.CV_16S,0,1)
    abs_sobelX, abs_sobelY = cv2.convertScaleAbs(sobelX), cv2.convertScaleAbs(sobelY)
    combined_sobel = cv2.addWeighted(abs_sobelX, 0.5, abs_sobelY, 0.5, 0)
    contours,_ = cv2.findContours(combined_sobel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours, abs_sobelX, abs_sobelY, combined_sobel

def find_rectangle(contours,height,weight,step=1):
    rectangles = []
    not_rectangles = []
    centers = []
    areas = []
    for i in range(len(contours)):
        # epsilon = \* cv2.arcLength(contours[i],True)
        approx = cv2.approxPolyDP(contours[i],20,True)
        corners = len(approx)
        mm = cv2.moments(contours[i])
        cx = int(mm['m10'] / mm['m00'])
        cy = int(mm['m01'] / mm['m00'])
        area = cv2.contourArea(contours[i])
        if step == 2:
            if cx < 30 or cx > (weight - 30) or cy < 25 or cy > (height - 25):
                continue
        if corners == 4 and area > 1000.0:
            rectangles.append(contours[i])
            centers.append((cx,cy))
            areas.append(area)
        elif corners > 4 and area > 50:
            not_rectangles.append(contours[i])
        else:
            if step == 2:
                if area > 600:
                    rectangles.append(contours[i])
                    centers.append((cx,cy))
                    areas.append(area)
        
    return rectangles, not_rectangles, centers, areas

def app(filename):
    print(filename)
    # 讀取圖片並轉成灰階
    image = cv2.imread(filename)
    IMAGE_HEIGHT = image.shape[0]
    IMAGE_WEIGHT = image.shape[1]
    # image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    imageR = image[:,:,-1]
    temp_imageR = copy.deepcopy(imageR)

    _, thr = cv2.threshold(temp_imageR,150,255,cv2.THRESH_BINARY)
    _, thr2 = cv2.threshold(temp_imageR,90,255,cv2.THRESH_BINARY)

    contours_abnormal,sobelX_abnormal,sobelY_abnormal,sobel_abnormal = get_contours(thr)
    contours_normal,_,_,_ = get_contours(thr2)

    abnormal_rectangles, abnormal_not_rectangles, abnormal_centers, abnormal_areas = find_rectangle(contours_abnormal,IMAGE_HEIGHT,IMAGE_WEIGHT)
    _,_,normal_centers, normal_areas = find_rectangle(contours_normal,IMAGE_HEIGHT,IMAGE_WEIGHT)

    draw_abnormal = copy.deepcopy(image)
    cv2.drawContours(draw_abnormal,contours_normal,-1,(0,255,0),4)
    cv2.drawContours(draw_abnormal,abnormal_rectangles,-1,(0,0,255),8)
    cv2.drawContours(draw_abnormal,abnormal_not_rectangles,-1,(0,255,255),4)
    print(len(contours_abnormal),len(contours_normal))
    _, thr3 = cv2.threshold(draw_abnormal[:,:,1],5,255,cv2.THRESH_BINARY)
    contours_final,_,_,_ = get_contours(thr3)
    final_rectangles, final_not_rectangles, final_centers, final_areas = find_rectangle(contours_final,IMAGE_HEIGHT,IMAGE_WEIGHT,2)
    print('final contour centers: ',final_centers)
    print('-----------------------------------------------')
    print('final contour areas: ',final_areas)

    final_result = copy.deepcopy(image)
    cv2.drawContours(final_result,final_rectangles,-1,(0,255,0),5)
    cv2.drawContours(final_result,abnormal_rectangles,-1,(255,255,255),5)

    plt.subplot(331),plt.imshow(cv2.merge([imageR,image[:,:,1],image[:,:,0]]))
    plt.title('Origin image'), plt.xticks([]), plt.yticks([])
    plt.subplot(332),plt.imshow(temp_imageR, cmap='gray')
    plt.title('Red channel'), plt.xticks([]), plt.yticks([])
    plt.subplot(333),plt.imshow(thr, cmap='gray')
    plt.title('Threshold'), plt.xticks([]), plt.yticks([])
    plt.subplot(334),plt.imshow(sobelX_abnormal, cmap='gray')
    plt.title('SobelX'), plt.xticks([]), plt.yticks([])
    plt.subplot(335),plt.imshow(sobelY_abnormal, cmap='gray')
    plt.title('SobelY'), plt.xticks([]), plt.yticks([])
    plt.subplot(336),plt.imshow(sobel_abnormal, cmap='gray')
    plt.title('Sobel_combined'), plt.xticks([]), plt.yticks([])
    plt.subplot(337),plt.imshow(cv2.merge([draw_abnormal[:,:,-1],draw_abnormal[:,:,1],draw_abnormal[:,:,0]]))
    plt.title('Abnormal boundary'), plt.xticks([]), plt.yticks([])
    plt.subplot(338),plt.imshow(thr3, cmap='gray')
    plt.title('Threshold3'), plt.xticks([]), plt.yticks([])
    plt.subplot(339),plt.imshow(cv2.merge([final_result[:,:,-1],final_result[:,:,1],final_result[:,:,0]]))
    plt.title('Final'), plt.xticks([]), plt.yticks([])
    # plt.show()
    plt.savefig(f'./output/Result_{filename}')

if __name__ == '__main__':
    filenames = glob(os.path.join('./','*.jpg'))
    filenames = [filename.split('\\')[-1] for filename in filenames]
    for filename in filenames:
        app(filename)
