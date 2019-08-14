import tensorflow as tf

# 定义神经网络结构相关的参数
IMAGE_SIZE=28   #输入图片尺寸
NUM_CHANNELS=1  #输入图片深度
CONV1_SIZE=5    #第一层卷积层过滤器尺寸
CONV1_DEEP=32   #第一层卷积层深度
CONV2_SIZE=5    #第二层卷积层过滤器尺寸
CONV2_DEEP=64   #第二层卷积层深度_
FC_SIZE=512     #全连接层节点数
NUM_LABELS=1    #输出层节点个数 

# 定义神经网络的前向传播过程
def inference(input_tesnor, train,regularizer):
    # 声明第一层卷积层神经网络的变量并完成前向传播过程
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable('weight',[CONV1_SIZE, CONV1_SIZE,NUM_CHANNELS,CONV1_DEEP],
                                 initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable('biases', [CONV1_DEEP],
                                 initializer=tf.constant_initializer(0.0))
        conv1=tf.nn.conv2d(input_tesnor,conv1_weights,strides=[1,1,1,1],padding='SAME')
        bias1=tf.nn.bias_add(conv1,conv1_biases)
        actived_conv1=tf.nn.relu(bias1)
    # 声明第二层池化层神经网络的变量并完成前向传播过程
    with tf.variable_scope('layer2-pool1'):
        pool1=tf.nn.max_pool(actived_conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    #第三层
    with tf.variable_scope('layer3-conv2'):  
        conv2_weights = tf.get_variable('weight',[CONV2_SIZE,CONV2_SIZE,CONV1_DEEP,CONV2_DEEP],
                                 initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable('biases', [CONV2_DEEP],
                                 initializer=tf.constant_initializer(0.0))
        conv2=tf.nn.conv2d(pool1,conv2_weights,strides=[1,1,1,1],padding='SAME')
        bias2=tf.nn.bias_add(conv2,conv2_biases)
        actived_conv2=tf.nn.relu(bias2)
    #第四层
    with tf.variable_scope('layer4-pool2'):
        pool2=tf.nn.max_pool(actived_conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    
    #池化层输出格式转换为全连接层输入格式
    pool2_shape=pool2.get_shape().as_list()
    nodes=pool2_shape[1]*pool2_shape[2]*pool2_shape[3]   #pool2_shape[0]为batch数
    reshaped=tf.reshape(pool2,[pool2_shape[0],nodes])
    
    #第五层
    with tf.variable_scope('layer5-fc1'):
        fc1_weights=tf.get_variable('weight',[nodes,FC_SIZE],
                                 initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer!=None:
            tf.add_to_collection('loss',regularizer(fc1_weights))
        fc1_biases = tf.get_variable('biases', [FC_SIZE],
                                 initializer=tf.constant_initializer(0.1))    
        fc1=tf.nn.relu(tf.matmul(reshaped,fc1_weights)+fc1_biases)
        if train:
            fc1=tf.nn.dropout(fc1,0.5)
    #第六层
    with tf.variable_scope('layer6-fc2'):
        fc2_weights=tf.get_variable('weight',[FC_SIZE,NUM_LABELS],\
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer!=None:
            tf.add_to_collection('loss',regularizer(fc2_weights))
        fc2_biases = tf.get_variable('biases', [NUM_LABELS],
                                 initializer=tf.constant_initializer(0.1))    
        fc2=tf.matmul(fc1,fc2_weights)+fc2_biases
    y=tf.sigmoid(fc2)
    return y
        
