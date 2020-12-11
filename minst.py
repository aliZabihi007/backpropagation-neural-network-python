import tensorflow as tf


# تابع برای انجام محاسبات رو به جلو
def MLP(x):
    w1 = tf.Variable(tf.compat.v1.random_uniform([784, 256]))
    b1 = tf.Variable(tf.zeros([256]))
    h1 = tf.nn.relu(tf.matmul(x, w1) + b1)

    w2 = tf.Variable(tf.compat.v1.random_uniform([256, 128]))
    b2 = tf.Variable(tf.zeros([128]))
    h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)

    w3 = tf.Variable(tf.compat.v1.random_uniform([128, 10]))
    b3 = tf.Variable(tf.zeros([10]))
    out = tf.matmul(h2, w3) + b3
    return out


#  شروع کد مورد اجرا
if __name__ == '__main__':
    (x_tr, y_tr), (x_te, y_te) = tf.keras.datasets.mnist.load_data()
    x_va = x_tr[50000:60000]
    x_tr = x_tr[0:50000]
    y_va = y_tr[50000:60000]
    y_tr = y_tr[0:50000]
    # تبدیل ماتریس به ارایه  قابل ورود و مورد برسی
    x_tr = x_tr.reshape(50000, 784)
    x_va = x_va.reshape(10000, 784)
    x_te = x_te.reshape(10000, 784)

    x_tr = x_tr.astype('float32')
    x_va = x_va.astype('float32')
    x_te = x_te.astype('float32')
    # محدود کردن در یک بازده مناسب برای وارسی
    grayescale = 255
    x_tr /= grayescale
    x_va /= grayescale
    x_te /= grayescale
    # ایجاد ده کلاس به عنوان داده های خروجی برای  برسی کردن
    num_class = 10
    y_tr = tf.keras.utils.to_categorical(y_tr, num_class)
    y_va = tf.keras.utils.to_categorical(y_va, num_class)
    y_te = tf.keras.utils.to_categorical(y_te, num_class)

    tf.compat.v1.disable_eager_execution()
    X = tf.compat.v1.placeholder(tf.float32, [None, 784])
    Y = tf.compat.v1.placeholder(tf.float32, [None, 10])
    outs = MLP(X)
    # محاسبه میزان خطا  نسبت به مقدار مورد نظر و مقدار واقعی
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=outs))
    train_op = tf.compat.v1.train.AdamOptimizer(0.01).minimize(loss_op)

    init = tf.compat.v1.global_variables_initializer()
    # انجام محاسبات در در 30 ایپاک و در 1000 بچ برای اعمال شدن
    epachcnt = 30
    batch = 1000
    itration = len(x_tr) // batch

    with tf.compat.v1.Session() as see:
        see.run(init)
        for epach in range(epachcnt):
            avg_loss = 0
            start = 0;
            end = batch

            for i in range(itration):
                _, loss = see.run([train_op, loss_op], feed_dict={X: x_tr[start:end], Y: y_tr[start:end]})
                start += batch;
                end += batch
                avg_loss += loss / itration

            pred = tf.nn.softmax(outs)
            correct = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
            accura = tf.reduce_mean(tf.cast(correct, "float"))
            cur_val_acc = accura.eval({X: x_va, Y: y_va})
            print("epock : " + str(epach) + " validation: " + str(cur_val_acc))
        pred = tf.nn.softmax(outs)
        correct = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))

        accura = tf.reduce_mean(tf.cast(correct, "float"))
        print("Test Accurecy: ", accura.eval({X: x_te, Y: y_te}))
        # print(accura.eval({X:}))
