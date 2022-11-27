import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def distance(point1,point2):
    #计算距离公式
    x1_distance = point1-point2
    x1_distance = x1_distance**2
    x1_distance = np.sum(x1_distance,axis=1)
    x1_distance = x1_distance**0.5
    return x1_distance

def visualize(image1,image2):
    #可视化
    fig = plt.figure()
    a1 = fig.add_subplot(121)
    a2 = fig.add_subplot(122)
    a1.imshow(image1)
    a2.imshow(image2)
    plt.show()

def classify(k,date_image):
    
    #初始化
    centroil_set = []
    distance_set = []
    label_set = []
    date_shape = []
    
    #图片数据预处理
    if (date_image.shape[2]>3):
        date_image = np.delete(date_image,3,2)
    date_shape =  date_image.shape
    date_image =date_image.reshape(-1,date_image.shape[2])

    #分类点与输出初始化
    for i in range (0,k):
        c = np.random.randint(len(date_image))
        centroil = date_image[c]
        centroil_set.append(centroil)
    after_date = np.array(date_image)

    #逼近分类
    for j in range(1,10):
        for centroil_2 in centroil_set:
            distance_set.append(distance(date_image,centroil_2))

        distance_set = np.array(distance_set).T

        label_set = np.argmin(distance_set,axis=1)

        for i in range(0,k):
            centroil_set[i] = np.mean(date_image[np.argwhere(label_set == i)],axis=0).reshape(3,)
        distance_set = []

    #分类后数据处理
    for i in range(0,k):
        after_date[np.argwhere(label_set == i)] = centroil_set[i]
    after_date = after_date.reshape(date_shape)
    return after_date 

path = input("pleasse input picture address:")
k=int(input("please input the number of species:"))    
date_before = mpimg.imread(path)
date_after = classify(k,date_before)
visualize(date_before,date_after)
print("finish!!!")
