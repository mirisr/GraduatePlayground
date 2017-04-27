import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import seaborn

from lab8_common import *

verbose = True

room_map = create_map()

state_keys = np.asmatrix([
    [ 0.1, 0.8, 0.87*2*np.pi ],
    [ 0.1, 0.8, np.pi ],
    [ 0.2, 0.6, np.pi ],
    [ 0.2, 0.2, np.pi ],
    [ 0.2, 0.2, np.pi/2 ],
    [ 0.22, 0.2, np.pi/2 ],
    [ 0.22, 0.2, 2*np.pi ],
    [ 0.22, 0.5, 2*np.pi ],
    [ 0.22, 0.5, np.pi/2 ],
    [ 0.8, 0.5, np.pi/2 ]
])

# calculate sequence of true state by interpolating between key frames
true_states = np.zeros((3,0))
for skind in range( 0, len(state_keys)-1 ):
    start = state_keys[skind]
    dest = state_keys[skind+1]
    tmp = vec_linspace( start, dest, num=25 )
    true_states = np.hstack(( true_states, tmp ))

# add a bit of noise to the true states
true_states += 0.001*np.random.randn( true_states.shape[0], true_states.shape[1] )

# now generate actual sensor data
data = np.zeros((11,0))
for t in range(0,true_states.shape[1]):
    if verbose:
        plt.figure( 1 )
        plt.clf()
        show_map( room_map )

    tmp = cast_rays( np.asarray( true_states[:,t] ), room_map, verbose=verbose )

    if verbose:
        plt.pause( 0.01 )
#        plt.savefig( "frames/%04d.png" % t )

    data = np.hstack(( data, np.atleast_2d(tmp).T ))
    print t

scipy.io.savemat( 'sensors.mat', { 'sonars':data, 'true_states':true_states } )
