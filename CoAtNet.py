import tensorflow as tf 
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Softmax,BatchNormalization, GlobalAveragePooling2D, MaxPool2D, Reshape,LayerNormalization, AveragePooling2D
from tensorflow.keras import Sequential
import numpy as np
import tensorflow_addons as tfa 


def conv_3x3_bn(inp, dim,image_size, downsample=False):
	stride = 1 if downsample == False else 2
	return Sequential([
		Conv2D(dim, 3, stride, padding='same', use_bias = False),
		BatchNormalization(),
		tfa.layers.GELU()])


class PreNorm(tf.keras.layers.Layer):
	def __init__(self,inp, fn, norm):
		super().__init__()
		self.norm = norm
		self.fn = fn

	def call(self, x):
		return self.fn(self.norm(x))


class Squeeze_excitation_layer(tf.keras.layers.Layer):
	def __init__(self, inp, out_dim, ratio = 4):
		super().__init__()
		self.out_dim = out_dim
		self.avg_pool = GlobalAveragePooling2D()
		self.fn = Sequential([
			Dense(int(inp/ratio)),
			tfa.layers.GELU(),
			Dense(out_dim,activation='sigmoid')
			])

	def call(self, x ):
		squeeze = self.avg_pool(x)
		excitation = tf.reshape(self.fn(squeeze),[-1, 1, 1, self.out_dim])
		out = tf.multiply(x,excitation)
		return out


class FeedForward(tf.keras.layers.Layer):
	def __init__(self, dim, hidden_dim, dropout=0.):
		super().__init__()
		self.net = Sequential([
			Dense(hidden_dim),
			tfa.layers.GELU(),
			Dropout(dropout),
			Dense(dim),
			Dropout(dropout)
			])

	def call(self,x):
		return self.net(x)


class MBConv(tf.keras.layers.Layer):
	def __init__(self, inp, oup, image_size, downsample = False, expansion = 4):
		super().__init__()
		self.downsample = downsample
		stride = 1 if self.downsample == False else 2
		hidden_dim = int(inp * expansion)

		if self.downsample:
			self.pool = MaxPool2D(pool_size=(3, 3), strides = 2, padding='same')
			self.proj = Conv2D(oup, 1, 1, padding='valid', use_bias=False)


		if expansion == 1:
			self.conv = Sequential([
				Conv2D(hidden_dim, 3, stride, padding='same', groups = hidden_dim, use_bias = False),
				BatchNormalization(),
				tfa.layers.GELU(),
				Conv2D(oup, 1 ,1, padding='valid', use_bias = False),
				BatchNormalization()
				])

		else: 
			self.conv = Sequential([
				Conv2D(hidden_dim, 1, stride, padding='valid', use_bias= False),
				BatchNormalization(),
				tfa.layers.GELU(),
				Conv2D(hidden_dim, 3, 1, padding='same', groups = hidden_dim, use_bias = False),
				BatchNormalization(),
				tfa.layers.GELU(),
				Squeeze_excitation_layer(inp, hidden_dim),
				Conv2D(oup, 1, 1, padding='valid', use_bias=False),
				BatchNormalization()
				])
		self.conv = PreNorm(inp, self.conv, BatchNormalization())


	def call(self, x):
		if self.downsample:
			return self.proj(self.pool(x)) + self.conv(x)
		else:
			return x + self.conv(x)


class MultiHeadAttention(tf.keras.layers.Layer):
	def __init__(self,inp, out, image_size, dim_head = 32, heads = 8, dropout=0.):
		super().__init__()
		project_out = not (heads == 1 and dim_head == inp)
		self.heads = heads
		self.dim_head = dim_head
		self.ih, self.iw = image_size
		inner_dim = dim_head *heads
		self.relative_table = tf.Variable(tf.zeros([(2*self.ih -1)*(2*self.iw-1),self.heads]))
		coords =  tf.meshgrid(tf.range(self.ih), tf.range(self.iw))
		coords = list(map(lambda t: tf.reshape(t,[1,self.ih*self.iw]), coords))
		coords = tf.concat(coords, axis =0)
		relative_coords = coords[:,:,None] - coords[:,None,:]
		relative_coords = np.array(relative_coords)
		relative_coords[0]  += self.ih - 1
		relative_coords[1]  += self.iw - 1
		relative_coords[0]  *=  2 * self.iw - 1
		relative_coords  = tf.convert_to_tensor(relative_coords)
		relative_coords  = tf.transpose(relative_coords,[1,2,0])
		self.relative_index  = tf.reshape(tf.reduce_sum(relative_coords, -1),-1)
		self.attend = Softmax(axis=-1)
		self.to_qkv = Dense(inner_dim*3, use_bias = False)
		self.out  = Sequential([
			Dense(out),
			Dropout(dropout)
			]) if project_out else tf.identity()
	
	def call(self, x):
		batch = tf.shape(x)[0]
		n = tf.shape(x)[1]
		qkv = tf.split(self.to_qkv(x),3, axis=-1)
		qkv = map(lambda t: tf.reshape(t, shape=[batch,n, self.heads, self.dim_head]),qkv)
		q, k, v = map(lambda t: tf.transpose(t, [0, 2, 1, 3]),qkv)
		dots = tf.matmul(q, k, transpose_b = True)
		relative_bias = tf.gather(self.relative_table,self.relative_index)
		relative_bias = tf.expand_dims(tf.reshape(relative_bias,[self.ih*self.iw, self.ih*self.iw, self.heads]),axis=0)
		relative_bias = tf.transpose(relative_bias, [0,3,2,1])
		dots = dots + relative_bias
		attention = self.attend(dots)
		out  = tf.matmul(attention,v)
		out = tf.reshape(tf.transpose(out,[0,2,1,3]),[batch,n,self.heads*self.dim_head])
		out = self.out(out)
		return out


class TransformerBlock(tf.keras.layers.Layer):
	def __init__(self,inp, out, image_size, heads = 8, dim_head = 32, downsample = False, dropout = .0):
		super().__init__()
		hidden_dim = int(inp * 4)
		self.ih, self.iw = image_size
		self.downsample = downsample

		if self.downsample:
			self.pool1 = MaxPool2D(pool_size=(3, 3), strides = 2, padding='same')
			self.pool2 = MaxPool2D(pool_size=(3, 3), strides = 2, padding='same')
			self.proj = Conv2D(out, 1, 1, padding = 'valid', use_bias = False)

		self.attn = MultiHeadAttention(inp, out, image_size, heads, dim_head, dropout)
		self.ff = FeedForward(out, hidden_dim, dropout)
		self.attn = Sequential([
			Reshape((self.ih*self.iw,-1)),
			PreNorm(inp,self.attn, LayerNormalization()),
			Reshape((self.ih,self.iw,-1))
			])
		self.ff = Sequential([
			Reshape((self.ih*self.iw,-1)),
			PreNorm(inp,self.ff, LayerNormalization()),
			Reshape((self.ih,self.iw,-1))
			])


	def call(self, x):
		if self.downsample:
			x = self.proj(self.pool1(x)) + self.attn(self.pool2(x))
		else:
			x = x + self.attn(x)
		x = x + self.ff(x)
		return x 



class CoAtNet(tf.keras.Model):
	def __init__(self, image_size, in_channels, num_blocks, channels, num_classes=1000, block_types=['C', 'C', 'T', 'T']):
		super().__init__()
		ih, iw = image_size
		block = {'C': MBConv, 'T': TransformerBlock}
		self.s0 = self._make_layer(conv_3x3_bn, in_channels, channels[0], num_blocks[0], (ih // 2, iw // 2))
		self.s1 = self._make_layer(block[block_types[0]], channels[0], channels[1], num_blocks[1], (ih // 4, iw // 4))
		self.s2 = self._make_layer(block[block_types[1]], channels[1], channels[2], num_blocks[2], (ih // 8, iw // 8))
		self.s3 = self._make_layer(block[block_types[2]], channels[2], channels[3], num_blocks[3], (ih // 16, iw // 16))
		self.s4 = self._make_layer(block[block_types[3]], channels[3], channels[4], num_blocks[4], (ih // 32, iw // 32))
		self.pool = AveragePooling2D(ih // 32, 1)
		self.faltten = Reshape((-1,))
		self.fc = Dense(num_classes, use_bias=False)
	def call(self, x):
		x = self.s0(x)
		x = self.s1(x)
		x = self.s2(x)
		x = self.s3(x)
		x = self.s4(x)
		x = self.pool(x)
		x = self.faltten(x)
		x = self.fc(x)
		return x
	def _make_layer(self, block, inp, oup, depth, image_size):
		layers = []
		for i in range(depth):
			if i == 0:
				layers.append(block(inp, oup, image_size, downsample=True))
			else:
				layers.append(block(oup, oup, image_size))
		return Sequential([*layers])






def coatnet_0():
    num_blocks = [2, 2, 3, 5, 2]           
    channels = [64, 96, 192, 384, 768]      
    return CoAtNet((224, 224), 3, num_blocks, channels, num_classes=1000)


def coatnet_1():
    num_blocks = [2, 2, 6, 14, 2]          
    channels = [64, 96, 192, 384, 768]     
    return CoAtNet((224, 224), 3, num_blocks, channels, num_classes=1000)


def coatnet_2():
    num_blocks = [2, 2, 6, 14, 2]           
    channels = [128, 128, 256, 512, 1026]   
    return CoAtNet((224, 224), 3, num_blocks, channels, num_classes=1000)


def coatnet_3():
    num_blocks = [2, 2, 6, 14, 2]           
    channels = [192, 192, 384, 768, 1536]   
    return CoAtNet((224, 224), 3, num_blocks, channels, num_classes=1000)


def coatnet_4():
    num_blocks = [2, 2, 12, 28, 2]          
    channels = [192, 192, 384, 768, 1536]  
    return CoAtNet((224, 224), 3, num_blocks, channels, num_classes=1000)




if __name__ == '__main__':
    img = tf.random.uniform([1, 224, 224, 3])
	
    net = coatnet_0()
    out = net(img)
    print(out.shape)

    # net = coatnet_1()
    # out = net(img)
    # print(out.shape)

    # net = coatnet_2()
    # out = net(img)
    # print(out.shape)

    # net = coatnet_3()
    # out = net(img)
    # print(out.shape)

    # net = coatnet_4()
    # out = net(img)
    # print(out.shape)

