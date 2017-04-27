import math
import pygame
from pygame.locals import *
import random as rand
import numpy as np
import sys
from numpy import atleast_2d
import pickle
from PIL import Image
import PIL.ImageOps
from tqdm import tqdm

# def load_data():
#     try:
#         with open("points_simple.dat") as f:
#             x = pickle.load(f)
#     except:
#         x = []
#     return x

points = []
X = []
Y = []

def save_data(data):
    with open("points_tri_2.dat", "wb") as f:
        pickle.dump(data, f)



def InitScreen(xdim, ydim):

	pygame.init()
	pygame.font.init()

	size = (xdim, ydim)
	screen = pygame.display.set_mode(size)

	pygame.display.set_caption("Create Points")
	clock = pygame.time.Clock()

	return screen, clock

'''
	Updates the pygame screen
	and allows for exiting of the pygame screen
'''

def Update():
	global points
	pygame.display.update()
	for e in pygame.event.get():
		if e.type == QUIT or (e.type == KEYUP and e.key == K_ESCAPE):
			#Change to nparrays
			x = np.zeros((len(X), 1))
			y = np.zeros((len(Y), 1))
			x[:,0] = X
			y[:,0] = Y
			points.append(x)
			points.append(y)
			save_data(np.array(points))

			sys.exit("Exiting")
		if e.type == MOUSEBUTTONDOWN:
		    return pygame.mouse.get_pos()


'''
	main function

'''

def main():
	'''
	xdim and ydim of the pygame screen 
	'''
	xdim = 500
	ydim = 500
	
	array = np.zeros([xdim, ydim])
	screen, clock = InitScreen(xdim, ydim)

	# Clear canvas
	screen.fill((255,255,255))

	s = pygame.Surface((xdim,ydim))  	# the size of your rect
	#s.set_alpha(0)                		# alpha level
	s.fill((255,255,255))           	# this fills the entire surface
	screen.blit(s, (0,0))

	#pygame.draw.circle(screen, (0,255,0), start_paint, 5)
	#pygame.draw.circle(screen, (255,0,0), end_paint, 5)

	
	
	myfont = pygame.font.SysFont("monospace", 15)
	while True:
		mouseClick = Update()
		if mouseClick != None:
			loc =  mouseClick[0], mouseClick[1]
			print loc,","
			pygame.draw.circle(screen, (0,0,0), mouseClick, 2)
			label = myfont.render(str(loc), 3, (0,0,0))
			label_loc = (mouseClick[0] -30, mouseClick[1] - 20)
			X.append(mouseClick[0])
			Y.append(mouseClick[1])
			#screen.blit(label,label_loc)
			
		pygame.time.delay(10)

		                    

if __name__ == '__main__':
    main()

