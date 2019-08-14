import numpy as np
import tensorflow as tf
import Inference    
import Convert_image_data 

#加载处理好之后的数据文件
INPUT_PICTURES=Convert_image_data.OUTPUT_FILE
INPUT_LABELS=r'E:\Work and Learning\tensorflow exercise(vscode)\Code\Defective_Part_Surface_Detecting_190813\Train_data\input_data\image and label\Labels.txt'

# 模型参数保存的路径和文件名
MODEL_SAVE_PATH = r'E:\Work and Learning\tensorflow exercise(vscode)\Code\Defective_Part_Surface_Detecting_190813\Train_data\parameter_save'
MODEL_NAME= r'\model.ckpt'

# 配置神经网络参数
BATCH_SIZE = 50        
LEARNING_RATE_BASE = 0.001    
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.0001
#TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99

def train():
    #加载预处理好的数据
    processed_data = np.load(INPUT_PICTURES,allow_pickle=True)  #写allow_pickle=True，否则显示Object arrays cannot be loaded when allow_pickle=False 
    processed_data_label=np.loadtxt(INPUT_LABELS,dtype='int32',delimiter=',')
    
    training_images = np.array(processed_data[0])   #训练数据  
    n_training_example = len(training_images)     #训练数据数量
    training_images_index = processed_data[1]     #训练数据对应的序号
    training_labels =processed_data_label[training_images_index,1]     #训练数据的标签
    
    validation_images = np.array(processed_data[2])   #验证数据
    n_validation_example = len(validation_images)
    validation_images_index = processed_data[3]
    validation_labels =processed_data_label[validation_images_index,1]

    print('%d training examples and %d validation examples'
          % (n_training_example, len(validation_labels)))
    
    # 定义输入输出placeholder
    x = tf.placeholder(tf.float32, [BATCH_SIZE,Inference.IMAGE_SIZE,Inference.IMAGE_SIZE,Inference.NUM_CHANNELS],
                       name='x-input')
    y_ = tf.placeholder(tf.float32, [BATCH_SIZE],    
                        name='y-input')
    
    #定义正则化操作
    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    
    # 使用Inference.py中定义的前向传播过程
    y = Inference.inference(x,True,regularizer)
    
    #定义训练步数
    global_step = tf.Variable(0, trainable=False)
    
    # 定义滑动平均操作
    variable_average = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    variable_average_op = variable_average.apply(tf.trainable_variables())
    
    #定义交叉熵损失
    cross_entropy=-tf.reduce_mean(y_*tf.log(tf.clip_by_value(y,1e-10,1.0))\
        +(1-y_)*tf.log(tf.clip_by_value(1-y,1e-10,1.0)))
    
    #定义包含正则化损失的总损失
    loss = cross_entropy + tf.add_n(tf.get_collection('loss'))   
                                                                      
    #定义正确率
    one = tf.ones_like(y)          
    zero = tf.zeros_like(y)
    logit = tf.where(y <0.5, x=zero, y=one)

    correct_prediction = tf.equal(logit, y_)  
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) #将True和False转换为浮点数，并取平均值，既为正确率
    
    #定义衰减学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE, global_step,
        n_training_example / BATCH_SIZE, LEARNING_RATE_DECAY,staircase=True)
    
    #定义训练操作
    train_step = tf.train.RMSPropOptimizer(learning_rate)\
            .minimize(loss, global_step=global_step)
    
    #反向传播更新参数后，再更新每个参数的滑动平均值
    with tf.control_dependencies([train_step, variable_average_op]):
        train_op = tf.no_op(name='train')
    
    # 初始化TensorFlow持久化类
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess,MODEL_SAVE_PATH+MODEL_NAME)
        else:
            sess.run(tf.global_variables_initializer())
        while 1:   
            #训练集中随机位置选取batch
            start=np.random.randint(n_training_example-BATCH_SIZE)
            end=start+BATCH_SIZE
            
            #使用滑动平均值来训练
            _, loss_value, step = sess.run([train_op, loss, global_step],
                                           feed_dict={x: training_images[start:end],    
                                            y_: training_labels[start:end]})      
            
            accuracy_value = sess.run(accuracy,
                                            feed_dict={x: training_images[start:end],    
                                            y_: training_labels[start:end]})
            #每一百步输出训练batch上损失
            if step % 100 == 0:
                print('After %d training step(s), loss on training batch is %g.'
                    % (step, loss_value))
            #每一百步输出训练batch上正确率
            if step % 100 == 0:
                print('After %d training step(s), accuracy on training batch is %g.\n'
                    % (step, accuracy_value))
            #每一百步保存一次数据  
            if step % 100 == 0:    
                saver.save(sess,MODEL_SAVE_PATH+MODEL_NAME)     
                          
def main(argv=None):     
    train()
   
if __name__ == '__main__':
    tf.app.run()


















