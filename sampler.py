from skimage import transform
from functools import partial
import numpy as np

RESOLUTION = 0.001
num_grids = int(1/RESOLUTION+0.5)

def generate_lut(img):
	density_img = transform.resize(img,(num_grids,num_grids))
	accumulation = 0
	cumulative_prob = []

	for y in xrange(num_grids-1,-1,-1):
		for x in xrange(num_grids-1,-1,-1):
			accumulation += density_img[y][x]
			cumulative_prob.append(accumulation)

	cumulative_prob = np.array(cumulative_prob)
	cumulative_prob /= accumulation

	get_point =  partial(np.interp, xp = cumulative_prob, fp = range(len(cumulative_prob)) )
	return get_point 

def sampler(get_point,N):
	samples = np.zeros((N,2))
	for i in xrange(N):
		seed = np.random.random()
		point = int(get_point(seed))
		samples[i][0] = point / num_grids
		samples[i][1] = point % num_grids
		print i,':',samples[i]
	return samples



xdata = []
ydata = []
def update(fr,get_point,ln):
	print fr
	seed = np.random.random()
	point = int(get_point(seed))
	ydata.append((point / num_grids)*1.0/num_grids)
	xdata.append((point % num_grids)*1.0/num_grids)

	print (point / num_grids)*1.0/num_grids,(point % num_grids)*1.0/num_grids
	ln.set_data(xdata,ydata)
	return ln,



if __name__ == '__main__':
	from skimage import io
	img = io.imread('batman.jpg',True)
	get_point = generate_lut(img)
	# samples = sampler(get_point,10000)

	from matplotlib import pyplot
	import matplotlib.animation as animation

	fig,(ax0,ax1) = pyplot.subplots(ncols=2,figsize=(9,4))
	fig.canvas.set_window_title('Test 2D sampler')
	ax0.imshow(img,cmap='gray')
	ax0.xaxis.set_major_locator(pyplot.NullLocator())
	ax0.yaxis.set_major_locator(pyplot.NullLocator())


	ax1.axis('equal')
	ax1.axis([0, 1, 0, 1])
	ln, = ax1.plot([], [], 'k,', animated = True)

	line_ani = animation.FuncAnimation(fig, update, 10000, fargs=(get_point, ln), repeat_delay=False,interval=1, blit=True)
	line_ani.save('batman2.mp4')
	pyplot.show()