import tensorflow as tf
import os
import numpy as np

from configuration import training_config,generator_config,discriminator_config
from dataloader import Gen_Data_loader,Dis_dataloader
from generator import Generator
from rollout import rollout
from discriminator import Discriminator
import pickle
from target_lstm import TARGET_LSTM
from utils import generate_samples,target_loss


# config_hardware = tf.ConfigProto()
# config_hardware.gpu_options.per_process_gpu_memory_fraction = 0.40
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

def main(unused_argv):
    config_train = training_config()
    config_gen = generator_config()
    config_dis = discriminator_config()

    np.random.seed(config_train.seed)

    assert config_train.start_token == 0
    gen_data_loader = Gen_Data_loader(config_gen.gen_batch_size)
    likelihood_data_loader = Gen_Data_loader(config_gen.gen_batch_size)
    dis_data_loader = Dis_dataloader(config_dis.dis_batch_size)

    generator = Generator(config=config_gen)
    generator.build()

    rollout_gen = rollout(config=config_gen)

    #Build target LSTM
    target_params = pickle.load(open('save/target_params.pkl','rb'),encoding='iso-8859-1')
    target_lstm = TARGET_LSTM(config=config_gen, params=target_params) # The oracle model


    # Build discriminator
    discriminator = Discriminator(config=config_dis)
    discriminator.build_discriminator()


    # Build optimizer op for pretraining
    pretrained_optimizer = tf.train.AdamOptimizer(config_train.gen_learning_rate)
    var_pretrained = [v for v in tf.trainable_variables() if 'teller' in v.name]
    gradients, variables = zip(
        *pretrained_optimizer.compute_gradients(generator.pretrained_loss, var_list=var_pretrained))
    gradients, _ = tf.clip_by_global_norm(gradients, config_train.grad_clip)
    gen_pre_update = pretrained_optimizer.apply_gradients(zip(gradients, variables))

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    generate_samples(sess,target_lstm,config_train.batch_size,config_train.generated_num,config_train.positive_file)
    gen_data_loader.create_batches(config_train.positive_file)

    log = open('save/experiment-log.txt','w')
    print('Start pre-training generator....')

    log.write('pre-training...\n')

    for epoch in range(config_train.pretrained_epoch_num):
        gen_data_loader.reset_pointer()
        for it in range(gen_data_loader.num_batch):
            batch = gen_data_loader.next_batch()
            _,g_loss = sess.run([gen_pre_update,generator.pretrained_loss],feed_dict={generator.input_seqs_pre:batch,
                                                                                      generator.input_seqs_mask:np.ones_like(batch)})

        if epoch % config_train.test_per_epoch == 0:
            #进行测试，通过Generator产生一批序列，
            generate_samples(sess,generator,config_train.batch_size,config_train.generated_num,config_train.eval_file)
            # 创建这批序列的data-loader
            likelihood_data_loader.create_batches(config_train.eval_file)
            # 使用oracle 计算 交叉熵损失nll
            test_loss = target_loss(sess,target_lstm,likelihood_data_loader)
            # 打印并写入日志
            print('pre-train ',epoch, ' test_loss ',test_loss)
            buffer = 'epoch:\t' + str(epoch) + '\tnll:\t' + str(test_loss) + '\n'
            log.write(buffer)


    print('Start pre-training discriminator...')
    for t in range(config_train.dis_update_time_pre):
        print("Times: " + str(t))
        generate_samples(sess,generator,config_train.batch_size,config_train.generated_num,config_train.negative_file)
        dis_data_loader.load_train_data(config_train.positive_file,config_train.negative_file)
        for _ in range(config_train.dis_update_time_pre):
            dis_data_loader.reset_pointer()
            for it in range(dis_data_loader.num_batch):
                x_batch,y_batch = dis_data_loader.next_batch()
                feed_dict = {
                    discriminator.input_x : x_batch,
                    discriminator.input_y : y_batch,
                    discriminator.dropout_keep_prob : config_dis.dis_dropout_keep_prob
                }
                _ = sess.run(discriminator.train_op,feed_dict)



    # Build optimizer op for adversarial training
    train_adv_opt = tf.train.AdamOptimizer(config_train.gen_learning_rate)
    gradients, variables = zip(*train_adv_opt.compute_gradients(generator.gen_loss_adv, var_list=var_pretrained))
    gradients, _ = tf.clip_by_global_norm(gradients, config_train.grad_clip)
    train_adv_update = train_adv_opt.apply_gradients(zip(gradients, variables))

    # Initialize global variables of optimizer for adversarial training
    uninitialized_var = [e for e in tf.global_variables() if e not in tf.trainable_variables()]
    init_vars_uninit_op = tf.variables_initializer(uninitialized_var)
    sess.run(init_vars_uninit_op)

    # Start adversarial training
    for total_batch in range(config_train.total_batch):
        for iter_gen in range(config_train.gen_update_time):
            samples = sess.run(generator.sample_word_list_reshpae)

            feed = {'pred_seq_rollout:0':samples}
            reward_rollout = []
            for iter_roll in range(config_train.rollout_num):
                rollout_list = sess.run(rollout_gen.sample_rollout_step,feed_dict=feed)
                # np.vstack 它是垂直（按照行顺序）的把数组给堆叠起来。
                rollout_list_stack = np.vstack(rollout_list)
                reward_rollout_seq = sess.run(discriminator.ypred_for_auc,feed_dict={
                    discriminator.input_x:rollout_list_stack,discriminator.dropout_keep_prob:1.0
                })
                reward_last_tok = sess.run(discriminator.ypred_for_auc,feed_dict={
                    discriminator.input_x:samples,discriminator.dropout_keep_prob:1.0
                })
                reward_allseq = np.concatenate((reward_rollout_seq,reward_last_tok),axis=0)[:,1]
                reward_tmp = []
                for r in range(config_gen.gen_batch_size):
                    reward_tmp.append(reward_allseq[range(r,config_gen.gen_batch_size * config_gen.sequence_length,config_gen.gen_batch_size)])

                reward_rollout.append(np.array(reward_tmp))
                rewards = np.sum(reward_rollout,axis = 0) / config_train.rollout_num
                _,gen_loss = sess.run([train_adv_update,generator.gen_loss_adv],feed_dict={generator.input_seqs_adv:samples,
                                                                                           generator.rewards:rewards})


        if total_batch % config_train.test_per_epoch == 0 or total_batch == config_train.total_batch - 1:
            generate_samples(sess, generator, config_train.batch_size, config_train.generated_num, config_train.eval_file)
            likelihood_data_loader.create_batches(config_train.eval_file)
            test_loss = target_loss(sess, target_lstm, likelihood_data_loader)
            buffer = 'epoch:\t' + str(total_batch) + '\tnll:\t' + str(test_loss) + '\n'
            print ('total_batch: ', total_batch, 'test_loss: ', test_loss)
            log.write(buffer)


        for _ in range(config_train.dis_update_time_adv):
            generate_samples(sess,generator,config_train.batch_size,config_train.generated_num,config_train.negative_file)
            dis_data_loader.load_train_data(config_train.positive_file,config_train.negative_file)

            for _ in range(config_train.dis_update_time_adv):
                dis_data_loader.reset_pointer()
                for it in range(dis_data_loader.num_batch):
                    x_batch,y_batch = dis_data_loader.next_batch()
                    feed = {
                        discriminator.input_x:x_batch,
                        discriminator.input_y:y_batch,
                        discriminator.dropout_keep_prob:config_dis.dis_dropout_keep_prob
                    }
                    _ = sess.run(discriminator.train_op,feed)

    log.close()

if __name__ == '__main__':
    tf.app.run()



