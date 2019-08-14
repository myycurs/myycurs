import tensorflow as tf
import numpy as np
import Inference 
import Train   


def evaluate():
    processed_data = np.load(Train.INPUT_PICTURES,allow_pickle=True)  #写allow_pickle=True，否则显示Object arrays cannot be loaded when allow_pickle=False 
    processed_data_label=np.loadtxt(Train.INPUT_LABELS,dtype='int32',delimiter=',')

    testing_images = np.array(processed_data[4])
    n_testing_example = len(testing_images)
    testing_images_index = processed_data[5]
    testing_labels =processed_data_label[testing_images_index,1]

    print('%d testing examples'% (n_testing_example))
    
    x = tf.placeholder(tf.float32, [n_testing_example,Inference.IMAGE_SIZE,Inference.IMAGE_SIZE,Inference.NUM_CHANNELS],
                    name='x-input')
    y_ = tf.placeholder(tf.float32, [n_testing_example],    
                    name='y-input')
    
    y = Inference.inference(x,False,None)
    
    #定义正确率
    one = tf.ones_like(y)          
    zero = tf.zeros_like(y)
    logit = tf.where(y <0.5, x=zero, y=one)

    correct_prediction = tf.equal(logit, y_)  
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) #将True和False转换为浮点数，并取平均值，既为正确率
    
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(Train.MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess,Train.MODEL_SAVE_PATH+Train.MODEL_NAME)
            accuracy_value = sess.run(accuracy,
                                            feed_dict={x: testing_images,    
                                            y_: testing_labels}) 
            print('accuracy value on test batch = %g'
                        % ( accuracy_value))
        else:
            print('No model!')
def main(argv=None):
    evaluate()

if __name__ == '__main__':
    tf.app.run()
