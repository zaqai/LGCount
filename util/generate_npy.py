#几何自适应核
import numpy as np
from scipy.ndimage import gaussian_filter
import scipy
import scipy.spatial
import os
from PIL import Image
from multiprocessing import Pool
import json

def compute_density_map(im, points):
    #根据图片大小生成全0矩阵
    h, w = im.shape[:2]
    im_density = np.zeros((h, w))
    
    #如果坐标点个数即人头数为1，则将该点对应的像素位置设置为最大值255，并返回
    if len(points) == 1:
        x1 = max(1, min(w, round(points[0][0])))
        y1 = max(1, min(h, round(points[0][1])))
        im_density[y1, x1] = 255
        return im_density
    
    # pts = np.array([[i,j] for i, j in zip(np.nonzero(im)[1], np.nonzero(im)[0])])
    # pts = np.array(zip(np.nonzero(im)[1], np.nonzero(im)[0]))
    pts = np.array(list(zip(np.nonzero(im)[1], np.nonzero(im)[0])))
    

    leafsize = 2048
    # build kdtree
    tree = scipy.spatial.KDTree(points.copy(), leafsize=leafsize)
    # query kdtree
    
 
    #对每个标注点开始遍历
    #定义一个高斯窗口的尺寸(f_sz)和标准差(sigma)，并使用fspecial函数创建一个高斯滤波器(H)
    for point in points:
        f_sz = 15
        # point = [int(i) for i in point]
        distances, locations = tree.query(point, k=4)
    #kd数搜索出离该目标点最近的4个标注点距离，去除最近的，剩余3个距离相加取平均，再乘上文章中取的beta=0.3
        sigma = (distances[1]+distances[2]+distances[3])*0.5/3
    #生成高斯核
        if sigma>300:
            sigma = 100
        f_sz = int(sigma)
        if f_sz%2==0:
            f_sz = f_sz-1
        H = gaussian_filter(np.ones((f_sz, f_sz)), sigma//4)
    #将坐标点坐标floor向下取整，计算出在矩阵中的位置，并限定在图像最大尺寸与【1，1】之间
        x = min(w, max(1, int(np.floor(point[0]))))
        y = min(h, max(1, int(np.floor(point[1]))))
    #超出范围的点舍弃
        if x > w or y > h:
            continue
    #根据窗口大小计算出左上角(x1,y1)和右下角坐标(x2,y2)
        x1 = x - int(f_sz/2)
        y1 = y - int(f_sz/2)
        x2 = x + int(f_sz/2)
        y2 = y + int(f_sz/2)
            
        dfx1 = 0
        dfy1 = 0
        dfx2 = 0
        dfy2 = 0
        change_H = False
    #判断高斯核尺寸是否超过图像边界如果超过则改变
        if x1 < 1:
            dfx1 = abs(x1) + 1
            x1 = 1
            change_H = True
        if y1 < 1:
            dfy1 = abs(y1) + 1
            y1 = 1
            change_H = True
        if x2 > w:
            dfx2 = x2 - w
            x2 = w
            change_H = True
        if y2 > h:
            dfy2 = y2 - h
            y2 = h
            change_H = True
    #计算裁剪后的区域在高斯滤波器核矩阵中的宽度和高度    
        x1h = 1 + dfx1
        y1h = 1 + dfy1
        x2h = f_sz - dfx2
        y2h = f_sz - dfy2
        
        if change_H:
            H = gaussian_filter(np.zeros((y2h - y1h + 1, x2h - x1h + 1)), sigma)
    #将高斯滤波器(H)加到矩阵 im_density 的相应位置上，以构建密度图
        im_density[y1-1:y2, x1-1:x2] += H
    
    return im_density


# 定义文件夹路径
img_path = r'/home/zhouqi/notebook/dataset/zsc-8k/images'
npy_path = r'/home/zhouqi/notebook/dataset/zsc-8k/gt_density'
anno_path = r'/home/zhouqi/notebook/ZSC-8k.json'
with open(anno_path) as f:
    anno = json.load(f)
    
# 定义处理图像的函数
def process_image(img_name):
    print(f'Processed: {img_name}')
    
    # 获取文件的完整路径
    file_path = os.path.join(img_path, img_name)
    image = Image.open(file_path)
    # 将图像转换为 NumPy 数组
    im = np.array(image)
    points = anno[img_name]['points']
    depthmap = compute_density_map(im, points)
    mask = depthmap > 0.0000
    mask = mask.astype(int) * 255
    np.save(f'{npy_path}/{img_name.split(".")[0]}.npy', mask)

# 创建进程池
with Pool(processes=32) as pool:
    # 遍历文件夹下的所有文件
    pool.map(process_image, os.listdir(img_path))