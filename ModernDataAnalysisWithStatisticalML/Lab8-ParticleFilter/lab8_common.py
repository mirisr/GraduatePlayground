import numpy as np
import matplotlib.pyplot as plt
import seaborn

def create_map():
    room_map = np.asarray([

        # outer walls
        [0, 0, 0, 1], 
        [0, 1, 1, 1],  
        [1, 1, 1, 0],  
        [1, 0, 0, 0],  

        # upper rooms
        [0,    0.6, 0.15, 0.6],  
        [0.25, 0.6, 0.6,  0.6],  
        [0.7,  0.6, 1.0,  0.6],
        [0.4,  1.0, 0.4,  0.6],

        # lower rooms
        [0,    0.4, 0.15, 0.4],  
        [0.25, 0.4, 0.8,  0.4],  
        [0.9,  0.4, 1.0,  0.4],
        [0.4,  0.0, 0.4,  0.4],

        # "table"
        [0.1, 0.1, 0.1, 0.15],  
        [0.1, 0.15, 0.15, 0.15],  
        [0.15, 0.15, 0.15, 0.1]
    ])

    return room_map

# adapted from the original matlab code at
# http://www.mathworks.com/matlabcentral/fileexchange/27205-fast-line-segment-intersection
def line_intersect( X1,Y1,X2,Y2, X3,Y3,X4,Y4 ):

    X4_X3 = X4.T - X3.T
    Y1_Y3 = Y1   - Y3.T
    Y4_Y3 = Y4.T - Y3.T
    X1_X3 = X1   - X3.T
    X2_X1 = X2   - X1
    Y2_Y1 = Y2   - Y1

    numerator_a = X4_X3 * Y1_Y3 - Y4_Y3 * X1_X3
    numerator_b = X2_X1 * Y1_Y3 - Y2_Y1 * X1_X3
    denominator = Y4_Y3 * X2_X1 - X4_X3 * Y2_Y1

    u_a = numerator_a / denominator
    u_b = numerator_b / denominator 

    INT_X = X1 + X2_X1 * u_a
    INT_Y = Y1 + Y2_Y1 * u_a
    INT_B = (u_a >= 0) & (u_a <= 1) & (u_b >= 0) & (u_b <= 1)

    return INT_X, INT_Y, INT_B

def show_map( room_map ):
    for i in range(0,room_map.shape[0]):
        plt.plot( [room_map[i,0], room_map[i,2]], [room_map[i,1], room_map[i,3]] )

    plt.axis('equal')
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])    

def vec_linspace( a, b, num=10 ):
    return np.asmatrix( [
        np.linspace( a[0,0], b[0,0], num=num ),
        np.linspace( a[0,1], b[0,1], num=num ),
        np.linspace( a[0,2], b[0,2], num=num )        
    ] )

def cast_rays( parts, room_map, verbose=False ):

    sensor_thetas = 0.1 * np.asarray( range(-5,6) )

    px = parts[0:1,:].T
    py = parts[1:2,:].T
    pt = parts[2:3,:].T

    rx1 = room_map[:,0:1]
    ry1 = room_map[:,1:2]
    rx2 = room_map[:,2:3]
    ry2 = room_map[:,3:4]

    dl = []
    for theta in sensor_thetas:
        # calculate intersections between all rays and all map lines
        int_x, int_y, int_b = line_intersect( px, py,
                                              px + np.sin(theta+pt),
                                              py + np.cos(theta+pt),
                                              rx1, ry1, rx2, ry2 )

        inds = int_b.ravel()

        # calculate nearest intersections
        # simplify code by setting non-intersections to be far away
        n_int_b = np.logical_not( int_b )
        int_x[ n_int_b ] = 1000 
        int_y[ n_int_b ] = 1000

        # calculate distances
        dx = px-int_x
        dy = py-int_y
        dists = dx*dx + dy*dy
        min_dists = np.min( dists, axis=1, keepdims=True )
        dl.append( min_dists )

        # show rays from a point, and their intersections
        # only needed for visualization
        if verbose:
            min_pts = np.argmin( dists, axis=1 )
            for i in range( 0, px.shape[0] ):
                plt.plot( [ px[i,0], int_x[i,min_pts[i]] ],
                          [ py[i,0], int_y[i,min_pts[i]] ] )

    data = np.hstack( dl )

    return data
