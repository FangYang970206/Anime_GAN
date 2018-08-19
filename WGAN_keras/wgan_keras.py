import os,scipy.misc
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Conv2D, ZeroPadding2D, BatchNormalization, Input
from keras.layers import Conv2DTranspose, Reshape, Activation, Cropping2D, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import RMSprop
from keras.activations import relu
from keras.initializers import RandomNormal
from keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm 
import numpy as np


conv_init = RandomNormal(0, 0.02)
gamma_init = RandomNormal(1., 0.02)

os.environ['KERAS_BACKEND']='tensorflow' 
os.environ['TENSORFLOW_FLAGS']='floatX=float32,device=cuda'

def DCGAN_D(isize, nc, ndf):
    inputs = Input(shape=(isize, isize, nc))
    x = ZeroPadding2D()(inputs)
    x = Conv2D(ndf, kernel_size=4, strides=2, use_bias=False, kernel_initializer=conv_init)(x)
    x = LeakyReLU(alpha=0.2)(x)
    for _ in range(4):
        x = ZeroPadding2D()(x)
        x = Conv2D(ndf*2, kernel_size=4, strides=2, use_bias=False, kernel_initializer=conv_init)(x)
        x = BatchNormalization(epsilon=1.01e-5, gamma_init=gamma_init)(x, training=1)
        x = LeakyReLU(alpha=0.2)(x)
        ndf *= 2    
    x = Conv2D(1, kernel_size=3, strides=1, use_bias=False, kernel_initializer=conv_init)(x)
    outputs = Flatten()(x)
    return Model(inputs=inputs, outputs=outputs)

def DCGAN_G(isize, nz, ngf):
    inputs = Input(shape=(nz,))
    x = Reshape((1, 1, nz))(inputs)
    x = Conv2DTranspose(filters=ngf, kernel_size=3, strides=2, use_bias=False,
                           kernel_initializer = conv_init)(x)
    for _ in range(4):
        x = Conv2DTranspose(filters=int(ngf/2), kernel_size=4, strides=2, use_bias=False,
                        kernel_initializer = conv_init)(x)
        x = Cropping2D(cropping=1)(x)
        x = BatchNormalization(epsilon=1.01e-5, gamma_init=gamma_init)(x, training=1) 
        x = Activation("relu")(x)
        ngf = int(ngf/2)
    x = Conv2DTranspose(filters=3, kernel_size=4, strides=2, use_bias=False,
                        kernel_initializer = conv_init)(x)
    x = Cropping2D(cropping=1)(x)
    outputs = Activation("tanh")(x)

    return Model(inputs=inputs, outputs=outputs)

nc = 3
nz = 100
ngf = 1024
ndf = 64
imageSize = 96
batchSize = 64
lrD = 0.00005 
lrG = 0.00005
clamp_lower, clamp_upper = -0.01, 0.01   

netD = DCGAN_D(imageSize, nc, ndf)
netD.summary()

netG = DCGAN_G(imageSize, nz, ngf)
netG.summary()

clamp_updates = [K.update(v, K.clip(v, clamp_lower, clamp_upper))
                          for v in netD.trainable_weights]
netD_clamp = K.function([],[], clamp_updates)

netD_real_input = Input(shape=(imageSize, imageSize, nc))
noisev = Input(shape=(nz,))

loss_real = K.mean(netD(netD_real_input))
loss_fake = K.mean(netD(netG(noisev)))
loss = loss_fake - loss_real 
training_updates = RMSprop(lr=lrD).get_updates(netD.trainable_weights,[], loss)
netD_train = K.function([netD_real_input, noisev],
                        [loss_real, loss_fake],    
                        training_updates)

loss = -loss_fake 
training_updates = RMSprop(lr=lrG).get_updates(netG.trainable_weights,[], loss)
netG_train = K.function([noisev], [loss], training_updates)

fixed_noise = np.random.normal(size=(batchSize, nz)).astype('float32')

datagen = ImageDataGenerator(
    # featurewise_center=True,
    # featurewise_std_normalization=True,
    rotation_range=20,
    rescale=1./255
)

train_generate = datagen.flow_from_directory("faces/", target_size=(96,96), batch_size=64, 
                                                shuffle=True, class_mode=None, save_format='jpg')

step = 0
print(dir(train_generate))
for step in range(100000):   
    
    for _ in range(5):
        real_data = (np.array(train_generate.next())*2-1)
        noise = np.random.normal(size=(batchSize, nz))
        errD_real, errD_fake  = netD_train([real_data, noise])
        errD = errD_real - errD_fake
        netD_clamp([])
    
    noise = np.random.normal(size=(batchSize, nz))  
    errG, = netG_train([noise])    
    print('[%d] Loss_D: %f Loss_G: %f Loss_D_real: %f Loss_D_fake %f' % (step, errD, errG, errD_real, errD_fake))
            
    if step%1000==0:
        netD.save("discriminator.h5")
        netG.save("generate.h5")
        fake = netG.predict(fixed_noise)
        display_grid = np.zeros((8*96,8*96,3))
        
        for j in range(int(64/8)):
            for k in range(int(64/8)):
                display_grid[j*96:(j+1)*96,k*96:(k+1)*96,:] = fake[k+8*j]
        img_save_path = os.path.join(os.getcwd(),"saved/img/{}.png".format(step))
        scipy.misc.imsave(img_save_path, display_grid)
