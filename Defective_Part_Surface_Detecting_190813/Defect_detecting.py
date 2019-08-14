import tensorflow as tf
import numpy as np
import Inference
import Train
import Convert_image_data

#加载处理好之后的数据文件
INPUT_PICTURES_detecting=Convert_image_data.OUTPUT_FILE_detecting

def detecting():
    processed_data = np.load(INPUT_PICTURES_detecting,allow_pickle=True)  #写allow_pickle=True，否则显示Object arrays cannot be loaded when allow_pickle=False 

    detecting_images = np.array(processed_data[0])
    n_detecting_example = len(detecting_images)
    
    x = tf.placeholder(tf.float32, [n_detecting_example,Inference.IMAGE_SIZE,Inference.IMAGE_SIZE,Inference.NUM_CHANNELS],
                    name='x-input')
  
    y = Inference.inference(x,False,None)
    
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(Train.MODEL_SAVE_PATH)   #判断是否有保存的模型
        
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess,Train.MODEL_SAVE_PATH+Train.MODEL_NAME)
            ys = sess.run(y,feed_dict={x: detecting_images})                            
            ys=np.array(ys)
            print('\ndetecting result is',ys)
            defect_collection=[]
            for i in range(len(ys)):
                if ys[i]>=[0.5]:
                    defect_collection.append(i+1)
            print('\nDefective image index is:')
            print(defect_collection)
            
        else:
            print('No model!')
def main(argv=None):
    detecting()

if __name__ == '__main__':
    tf.app.run()
