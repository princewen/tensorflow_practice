import tensorflow as tf
import pandas as pd
import numpy as np
import os


input_x_size = 20
field_size = 2

vector_dimension = 3

total_plan_train_steps = 1000
# 使用SGD，每一个样本进行依次梯度下降，更新参数
batch_size = 1

all_data_size = 1000

lr = 0.01

MODEL_SAVE_PATH = "TFModel"
MODEL_NAME = "FFM"

def createTwoDimensionWeight(input_x_size,field_size,vector_dimension):
    weights = tf.truncated_normal([input_x_size,field_size,vector_dimension])

    tf_weights = tf.Variable(weights)

    return tf_weights

def createOneDimensionWeight(input_x_size):
    weights = tf.truncated_normal([input_x_size])
    tf_weights = tf.Variable(weights)
    return tf_weights

def createZeroDimensionWeight():
    weights = tf.truncated_normal([1])
    tf_weights = tf.Variable(weights)
    return tf_weights

def inference(input_x,input_x_field,zeroWeights,oneDimWeights,thirdWeight):
    """计算回归模型输出的值"""

    secondValue = tf.reduce_sum(tf.multiply(oneDimWeights,input_x,name='secondValue'))

    firstTwoValue = tf.add(zeroWeights, secondValue, name="firstTwoValue")


    thirdValue = tf.Variable(0.0,dtype=tf.float32)
    input_shape = input_x_size

    for i in range(input_shape):
        featureIndex1 = i
        fieldIndex1 = int(input_x_field[i])
        for j in range(i+1,input_shape):
            featureIndex2 = j
            fieldIndex2 = int(input_x_field[j])
            vectorLeft = tf.convert_to_tensor([[featureIndex1,fieldIndex2,i] for i in range(vector_dimension)])
            weightLeft = tf.gather_nd(thirdWeight,vectorLeft)
            weightLeftAfterCut = tf.squeeze(weightLeft)

            vectorRight = tf.convert_to_tensor([[featureIndex2,fieldIndex1,i] for i in range(vector_dimension)])
            weightRight = tf.gather_nd(thirdWeight,vectorRight)
            weightRightAfterCut = tf.squeeze(weightRight)

            tempValue = tf.reduce_sum(tf.multiply(weightLeftAfterCut,weightRightAfterCut))

            indices2 = [i]
            indices3 = [j]

            xi = tf.squeeze(tf.gather_nd(input_x, indices2))
            xj = tf.squeeze(tf.gather_nd(input_x, indices3))

            product = tf.reduce_sum(tf.multiply(xi, xj))

            secondItemVal = tf.multiply(tempValue, product)

            tf.assign(thirdValue, tf.add(thirdValue, secondItemVal))

    return tf.add(firstTwoValue,thirdValue)

def gen_data():
    labels = [-1,1]
    y = [np.random.choice(labels,1)[0] for _ in range(all_data_size)]
    x_field = [i // 10 for i in range(input_x_size)]
    x = np.random.randint(0,2,size=(all_data_size,input_x_size))
    return x,y,x_field


if __name__ == '__main__':
    global_step = tf.Variable(0,trainable=False)
    trainx,trainy,trainx_field = gen_data()
    #
    input_x = tf.placeholder(tf.float32,[input_x_size ])
    input_y = tf.placeholder(tf.float32)
    #


    lambda_w = tf.constant(0.001, name='lambda_w')
    lambda_v = tf.constant(0.001, name='lambda_v')

    zeroWeights = createZeroDimensionWeight()

    oneDimWeights = createOneDimensionWeight(input_x_size)

    thirdWeight = createTwoDimensionWeight(input_x_size,  # 创建二次项的权重变量
                                           field_size,
                                           vector_dimension)  # n * f * k

    y_ = inference(input_x, trainx_field,zeroWeights,oneDimWeights,thirdWeight)

    l2_norm = tf.reduce_sum(
        tf.add(
            tf.multiply(lambda_w, tf.pow(oneDimWeights, 2)),
            tf.reduce_sum(tf.multiply(lambda_v, tf.pow(thirdWeight, 2)),axis=[1,2])
        )
    )

    loss = tf.log(1 + tf.exp(input_y * y_)) + l2_norm

    train_step = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(total_plan_train_steps):
            for t in range(all_data_size):
                input_x_batch = trainx[t]
                input_y_batch = trainy[t]
                print(input_x_batch,input_y_batch)
                predict_loss,_, steps = sess.run([loss,train_step, global_step],
                                               feed_dict={input_x: input_x_batch, input_y: input_y_batch})

                print("After  {step} training   step(s)   ,   loss    on    training    batch   is  {predict_loss} "
                      .format(step=steps, predict_loss=predict_loss))

                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=steps)
                writer = tf.summary.FileWriter(os.path.join(MODEL_SAVE_PATH, MODEL_NAME), tf.get_default_graph())
                writer.close()
        #

















