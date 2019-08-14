import glob
import os.path
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile  
import Inference

INPUT_DATA= r'E:\Work and Learning\tensorflow exercise(vscode)\Code\Defective_Part_Surface_Detecting_190813\Train_data\input_data'
INPUT_DATA_detecting=r'E:\Work and Learning\tensorflow exercise(vscode)\Code\Defective_Part_Surface_Detecting_190813\Detecting_data\input_data'

OUTPUT_FILE = r'E:\Work and Learning\tensorflow exercise(vscode)\Code\Defective_Part_Surface_Detecting_190813\Train_data\output_file\processed_data.npy'
OUTPUT_FILE_detecting=r'E:\Work and Learning\tensorflow exercise(vscode)\Code\Defective_Part_Surface_Detecting_190813\Detecting_data\output_file\processed_data_detecting.npy'

VALIDATION_PERCENTAGE = 0   #验证集占比
TEST_PERCENTAGE = 10         #测试集占比

VALIDATION_PERCENTAGE_detecting = 0     
TEST_PERCENTAGE_detecting = 0

def create_image_list(sess, input_data,testing_percentage, validation_percentage):
    sub_dirs = [x[0] for x in os.walk(input_data)]
    is_root_dir = True

    # 初始化各个数据集
    training_images = []     #训练图片数据集
    training_labels = []     #训练图片编号集
    testing_images = []
    testing_labels = []
    validation_images = []
    validation_labels = []
    current_label = 0

    # 读取所有的子目录
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue
        # 获取一个子目录中所有的图片文件
        extensions = ['jpg', 'jpeg', 'PNG']  
        file_list = []
        dir_name = os.path.basename(sub_dir)
        for extension in extensions:
            file_glob = os.path.join(input_data, dir_name, '*.' + extension)
            file_list.extend(glob.glob(file_glob))
        if not file_list: continue
        print('processing:', dir_name)
        i = 0
        # 处理图片数据
        for file_name in file_list:
            i += 1
            # 读取并解析图片，转换图片大小
            image_raw_data = gfile.FastGFile(file_name, 'rb').read()    # 读取图像
            image = tf.image.decode_jpeg(image_raw_data)                # 图像解码
            image=tf.image.rgb_to_grayscale(image)                      # 转为灰度图
            if image.dtype != tf.float32:
                image = tf.image.convert_image_dtype(image, dtype=tf.float32)       # 改变图像数据类型
            image = tf.image.resize_images(image, \
                [Inference.IMAGE_SIZE,Inference.IMAGE_SIZE])             #转换图片大小
            image_value = sess.run(image)

            # 随机划分数据集
            chance = np.random.randint(100)                #随机返回[0,100)的整数值
            if chance < validation_percentage:             #将部分图片加入验证集
                validation_images.append(image_value)      
                validation_labels.append(current_label)    #图片的编号
            elif chance < (testing_percentage + validation_percentage):     #将部分图片加入测试集
                testing_images.append(image_value)         
                testing_labels.append(current_label)
            else:                                          #将部分图片加入训练集
                training_images.append(image_value)       
                training_labels.append(current_label)
            if i % 1 == 0:
                print(i, 'images processed')
            current_label += 1
    
    # 划分完数据集后将训练数据随机打乱以获得更好的训练效果
    state = np.random.get_state()
    np.random.shuffle(training_images)
    np.random.set_state(state)
    np.random.shuffle(training_labels)    

    print('finish process data')
    return np.asarray([training_images, training_labels, validation_images, validation_labels,
                       testing_images, testing_labels])

# 数据整理主函数
def main():
    #选择要转换的图片是训练数据还是检测数据
    is_train_data=input('input 1 for train data | input 0 for detecting data: ')
    
    with tf.Session() as sess:
        if is_train_data=='1':
            processed_data = create_image_list(sess, INPUT_DATA,TEST_PERCENTAGE, VALIDATION_PERCENTAGE)
            print('Prepare write train data to file')
            np.save(OUTPUT_FILE, processed_data)
            print('Finish write train data to file')
        elif is_train_data=='0':
            processed_data = create_image_list(sess, INPUT_DATA_detecting,VALIDATION_PERCENTAGE_detecting, TEST_PERCENTAGE_detecting)
            print('Prepare write detecting data to file')
            np.save(OUTPUT_FILE_detecting, processed_data)
            print('Finish write detecting to file')
        else:
            print('input invalid!')

if __name__ == '__main__':
    main()
