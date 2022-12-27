# M11015Q02 柯元豪
import cv2
from glob import glob
import os
import copy
from matplotlib import pyplot as plt
from config import *
import pandas as pd

def get_contours(image):
    # 透過 Sobel operator 去找 x, y 的邊緣
    sobelX, sobelY = cv2.Sobel(image,cv2.CV_16S,1,0), cv2.Sobel(image,cv2.CV_16S,0,1)
    abs_sobelX, abs_sobelY = cv2.convertScaleAbs(sobelX), cv2.convertScaleAbs(sobelY)
    # 將 sobel x, sobel y 組在一起 (這裡採用相同權重)，取得各零件的輪廓
    combined_sobel = cv2.addWeighted(abs_sobelX, 0.5, abs_sobelY, 0.5, 0)
    # 找 contours
    contours,_ = cv2.findContours(combined_sobel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours, abs_sobelX, abs_sobelY, combined_sobel

def find_rectangle(contours,height,weight,step=1):
    rectangles = []
    not_rectangles = []
    centers = []
    areas = []
    for i in range(len(contours)):
        # 透過調整 epsilon 使 contour 近似成更少點組成的形狀 (這裡希望所有矩形元件，輪廓都能近似矩形)
        approx = cv2.approxPolyDP(contours[i],EPSILON,True)
        # 判斷該 contour 有幾個點
        corners = len(approx)
        # 取得中心點位置
        mm = cv2.moments(contours[i])
        cx = int(mm['m10'] / mm['m00'])
        cy = int(mm['m01'] / mm['m00'])
        # 取得 contour 面積
        area = cv2.contourArea(contours[i])
        # step 2 進一步決定瑕疵元件時，判斷該元件中心點是不是太靠近圖片邊界，如果是則不列為檢測對象
        if step == 2:
            if cx < NEAR_BOUNDARY_THR_X or cx > (weight - NEAR_BOUNDARY_THR_X) or cy < NEAR_BOUNDARY_THR_Y or cy > (height - NEAR_BOUNDARY_THR_Y):
                continue
        # 若 corner 數為四則將其視為矩形, 這裡額外設個面積的 threshold, 去排除掉過小的矩形 (可能是 noise 的 contour)
        if corners == 4 and area > RECTANGLE_AREA_THR:
            rectangles.append(contours[i])
            centers.append((cx,cy))
            areas.append(area)
        # 若 corner > 4 且面積有大於閥值, 則判斷為可能是瑕疵的元件
        elif corners > 4 and area > NOT_RECTANGLE_AREA_THR:
            not_rectangles.append(contours[i])
        # 在 step 2 進一步分析時，若依然存在面積夠大且又不是矩形的元件，而其中心點位置又在分析範圍內的話，依然將其視為瑕疵元件
        else:
            if step == 2:
                if area > STEP2_CONTOUR_AREA_THR:
                    rectangles.append(contours[i])
                    centers.append((cx,cy))
                    areas.append(area)
    return rectangles, not_rectangles, centers, areas

def app(filename):
    print('Start detecting: ',filename)
    # 讀檔
    image = cv2.imread(filename)
    IMAGE_HEIGHT = image.shape[0]
    IMAGE_WEIGHT = image.shape[1]
    # 取紅色通道的值
    imageR = image[:,:,-1]
    temp_imageR = copy.deepcopy(imageR)

    # 根據兩個不同閥值取得二值化結果
    _, thr = cv2.threshold(temp_imageR,THRESHOLD1,255,cv2.THRESH_BINARY)
    _, thr2 = cv2.threshold(temp_imageR,THRESHOLD2,255,cv2.THRESH_BINARY)

    # 取得輪廓資訊
    contours_thr1,sobelX_thr1,sobelY_thr1,sobel_thr1 = get_contours(thr)
    contours_thr2,_,_,_ = get_contours(thr2)

    # 對 thr1 的輪廓進一步分析，找出矩形與非矩形元件
    thr1_rectangles, thr1_not_rectangles, _, _ = find_rectangle(contours_thr1,IMAGE_HEIGHT,IMAGE_WEIGHT)

    # 將輪廓資訊疊回原圖，後續做第二步分析
    draw_thr1 = copy.deepcopy(image)
    # thr2 的輪廓疊綠色
    cv2.drawContours(draw_thr1,contours_thr2,-1,(0,255,0),4)
    # thr1 矩形元件的輪廓疊紅色 (粗一點是為了更明顯的蓋掉 thr2)
    cv2.drawContours(draw_thr1,thr1_rectangles,-1,(0,0,255),8)
    # thr1 非矩形元件的輪廓疊黃色 (為了第二步分析也能從綠色通道取得數值)
    cv2.drawContours(draw_thr1,thr1_not_rectangles,-1,(0,255,255),4)

    ##########       Step 2       ###########

    # 根據閥值取得二值化結果
    _, thr_step2 = cv2.threshold(draw_thr1[:,:,1],STEP2_THRESHOLD,255,cv2.THRESH_BINARY)
    # 取得輪廓資訊
    contours_abnormal,_,_,_ = get_contours(thr_step2)
    # 增加一些條件，去找出異常元件
    abnormal_rectangles, _, abnormal_centers, abnormal_areas = find_rectangle(contours_abnormal,IMAGE_HEIGHT,IMAGE_WEIGHT,2)
    
    ### For debug
    # print('abnormal contour centers: ',abnormal_centers)
    # print('-----------------------------------------------')
    # print('abnormal contour areas: ',abnormal_areas)

    # 最後疊圖 (圖片中沒框起來的則是邊緣較難以判斷的元件)
    abnormal_result = copy.deepcopy(image)
    # 綠框為異常元件
    cv2.drawContours(abnormal_result,abnormal_rectangles,-1,(0,255,0),5)
    # 白框為正常元件
    cv2.drawContours(abnormal_result,thr1_rectangles,-1,(255,255,255),5)

    # 以下畫圖、存檔
    plt.subplot(331),plt.imshow(temp_imageR, cmap='gray')
    plt.title('Red channel'), plt.xticks([]), plt.yticks([])
    plt.subplot(332),plt.imshow(thr, cmap='gray')
    plt.title('Threshold 1'), plt.xticks([]), plt.yticks([])
    plt.subplot(333),plt.imshow(thr2, cmap='gray')
    plt.title('Threshold 2'), plt.xticks([]), plt.yticks([])
    plt.subplot(334),plt.imshow(sobelX_thr1, cmap='gray')
    plt.title('SobelX_thr1'), plt.xticks([]), plt.yticks([])
    plt.subplot(335),plt.imshow(sobelY_thr1, cmap='gray')
    plt.title('SobelY_thr1'), plt.xticks([]), plt.yticks([])
    plt.subplot(336),plt.imshow(sobel_thr1, cmap='gray')
    plt.title('Sobel_combined_thr1'), plt.xticks([]), plt.yticks([])
    plt.subplot(337),plt.imshow(cv2.merge([draw_thr1[:,:,-1],draw_thr1[:,:,1],draw_thr1[:,:,0]]))
    plt.title('Step1 overlap'), plt.xticks([]), plt.yticks([])
    plt.subplot(338),plt.imshow(thr_step2, cmap='gray')
    plt.title('Threshold 3'), plt.xticks([]), plt.yticks([])
    plt.subplot(339),plt.imshow(cv2.merge([abnormal_result[:,:,-1],abnormal_result[:,:,1],abnormal_result[:,:,0]]))
    plt.title('Final Result'), plt.xticks([]), plt.yticks([])
    filename = filename.split('/')[-1].split('\\')[-1]
    plt.savefig(f'{ANALYSIS_DIR}/analysis_{filename}')
    cv2.imwrite(f'{RESULT_DIR}/result_{filename}',abnormal_result)
    return filename, abnormal_centers, abnormal_areas

if __name__ == '__main__':
    if not os.path.exists(ANALYSIS_DIR):
        os.makedirs(ANALYSIS_DIR)
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)
    filenames = glob(os.path.join(DATA_DIR,'*.jpg'))
    images = []
    exist_abnormal = []
    abnormals = []
    centers = []
    areas = []
    # 批次處理圖片 最後輸出兩張表
    for filename in filenames:
        filename, abnormal_centers, abnormal_areas = app(filename)
        images.append(filename)
        if len(abnormal_centers) == 0:
            exist_abnormal.append(False)
        else:
            exist_abnormal.append(True)
            for i in range(len(abnormal_centers)):
                image_name = filename.split('.')[0]
                abnormals.append(f'{image_name}_{i}')
                centers.append(abnormal_centers[i])
                areas.append(abnormal_areas[i])
    detection_df = pd.DataFrame({'file':images, 'exist abnormal':exist_abnormal})
    abnormals_df = pd.DataFrame({'abnormal_idx':abnormals, 'center': centers, 'area': areas})
    detection_df.to_csv(RESULT_CSV,index=False)
    abnormals_df.to_csv(ABNORMAL_LOC_CSV,index=False)