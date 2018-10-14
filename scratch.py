from keras.optimizers import SGD

op = SGD(lr=0.01)
a = op.get_config()
print(a)
