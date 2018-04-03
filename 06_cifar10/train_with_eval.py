FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('epoch', 30, "訓練するEpoch数")
tf.app.flags.DEFINE_string('data_dir', './data/', "訓練データのディレクトリ")
tf.app.flags.DEFINE_string('checkpoint_dir', './checkpoints/',
                          "チェックポイントを保存するディレクトリ")
tf.app.flags.DEFINE_string('test_data', None, "テストデータのパス")

def main(argv=None):
  global_step = tf.Variable(0, trainable=False)

  train_placeholder = tf.placeholder(tf.float32,
                                     shape=[32, 32, 3],
                                     name='input_image')
  label_placeholder = tf.placeholder(tf.int32, shape=[1], name='label')

  # (width, height, depth) -> (batch, width, height, depth)
  image_node = tf.expand_dims(train_placeholder, 0)

  logits = model.inference(image_node)
  total_loss = _loss(logits, label_placeholder)
  train_op = _train(total_loss, global_step)

  top_k_op = tf.nn.in_top_k(logits, label_placeholder, 1)

  with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    total_duration = 0

    for epoch in range(1, FLAGS.epoch + 1):
      start_time = time.time()

      for file_index in range(5):
        print('Epoch %d: %s' % (epoch, filenames[file_index]))
        reader = Cifar10Reader(filenames[file_index])

        for index in range(10000):
          image = reader.read(index)

          _, loss_value = sess.run([train_op, total_loss],
                                   feed_dict={
                                     train_placeholder: image.byte_array,
                                     label_placeholder: image.label
                                   })

          assert not np.isnan(loss_value), \
            'Model diverged with loss = NaN'

        reader.close()

      duration = time.time() - start_time
      total_duration += duration

      prediction = _eval(sess, top_k_op,
                         train_placeholder, label_placeholder)
      print('epoch %d duration = %d sec, prediction = %.3f'
            % (epoch, duration, prediction))

      tf.train.SummaryWriter(FLAGS.checkpoint_dir, sess.graph)

    print('Total duration = %d sec' % total_duration)
