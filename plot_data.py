# takes data saved by DRAW model and generates animations
# example usage: python plot_data.py noattn /tmp/draw/draw_data.npy

import matplotlib
import sys
import numpy as np

interactive=False # set to False if you want to write images to file

if not interactive:
	matplotlib.use('Agg') # Force matplotlib to not use any Xwindows backend.
import matplotlib.pyplot as plt

def xrecons_grid(X):
	"""
	plots grid of canvases for a single time step
	X is x_recons, (batch_size x img_size)
	assumes features = HxW images
	batch is assumed to be a square number
	"""
	H = C.shape[2]
	W = C.shape[3]
	padsize=1
	padval=.5
	ph=H+2*padsize # Padded height
	pw=W+2*padsize # Padded width
	batch_size=X.shape[0]
	# Take square root, because the image is going to contain a square grid of
	# reconstruction examples:
	N=int(np.sqrt(batch_size))
	X=X.reshape((N,N,H,W))
	img=np.ones((N*ph,N*pw))*padval
	for i in range(N):
		for j in range(N):
			startr=i*ph+padsize
			endr=startr+H
			startc=j*pw+padsize
			endc=startc+W
			img[startr:endr,startc:endc]=X[i,j,:,:]
	return img

if __name__ == '__main__':
	prefix=sys.argv[1]
	out_file=sys.argv[2]
	[C,Lxs,Lzs]=np.load(out_file)
	T,batch_size,H,W=C.shape
	X=1.0/(1.0+np.exp(-C)) # x_recons=sigmoid(canvas)
	if interactive:
		f,arr=plt.subplots(1,T)
	for t in range(T):
		img=xrecons_grid(X[t,:,:,:])
		if interactive:
			arr[t].matshow(img,cmap=plt.cm.gray)
			arr[t].set_xticks([])
			arr[t].set_yticks([])
		else:
			plt.matshow(img,cmap=plt.cm.gray)
			imgname='%s_%d.png' % (prefix,t) # you can merge using imagemagick, i.e. convert -delay 10 -loop 0 *.png mnist.gif
			plt.savefig(imgname)
			print(imgname)
	f=plt.figure()
	plt.plot(Lxs,label='Reconstruction Loss Lx')
	plt.plot(Lzs,label='Latent Loss Lz')
	plt.xlabel('iterations')
	plt.legend()
	if interactive:
		plt.show()
	else:
		plt.savefig('%s_loss.png' % (prefix))
