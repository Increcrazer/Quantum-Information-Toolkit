import numpy as np
from matplotlib import pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from plottool import set_axes_equal, Arrow3D

# Define parameter
pi = np.pi
cos = np.cos
sin = np.sin
arrow_prop_dict = dict(mutation_scale=20, arrowstyle='->', shrinkA=0, shrinkB=0)

# Create a sphere
def Create_sphere(fig):
    r = 1
    phi, theta = np.mgrid[0.0:pi:100j, 0.0:2.0*pi:100j]
    x = r*sin(phi)*cos(theta)
    y = r*sin(phi)*sin(theta)
    z = r*cos(phi) 

    ax = Axes3D(fig)
    ax.plot_surface(x, y, z, rstride=1, cstride=1, color='w', alpha=0.2, linewidth=0)

    # arrow_prop_dict = dict(mutation_scale=20, arrowstyle='->', shrinkA=0, shrinkB=0)
    a = Arrow3D([0, 1], [0, 0], [0, 0], **arrow_prop_dict, color='r')
    ax.add_artist(a)
    a = Arrow3D([0, 0], [0, 1], [0, 0], **arrow_prop_dict, color='b')
    ax.add_artist(a)
    a = Arrow3D([0, 0], [0, 0], [0, 1], **arrow_prop_dict, color='g')
    ax.add_artist(a)
    ax.text(1.1, 0, 0, r'$x$')
    ax.text(0, 1.1, 0, r'$y$')
    ax.text(0, 0, 1.1, r'$z$')

    ax.text(0, 0, 1.4, r'$|0\rangle$')
    ax.text(0, 0, -1.4, r'$|1\rangle$')

    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.zaxis.set_major_locator(MultipleLocator(0.5))
    return ax
 

def Bloch(r, theta, phi, ax, name):
    alpha = cos(theta/2)
    beta = np.exp(complex(0,phi)) *sin(theta/2)
    a = np.array([alpha, beta])
    # get quantum state location on Bloch sqhere
    qu_x = r*sin(theta)*cos(phi)
    qu_y = r*sin(theta)*sin(phi)
    qu_z = r*cos(theta)
     
    ax.scatter(qu_x,qu_y,qu_z,color="k",s=20)
    # arrow_prop_dict = dict(mutation_scale=20, arrowstyle='->', shrinkA=0, shrinkB=0)
    a = Arrow3D([0, qu_x], [0, qu_y], [0, qu_z], **arrow_prop_dict, color='black')
    ax.add_artist(a)
    ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1, 1, 4/3, 1.2]))
    ax.text(qu_x + 0.1, qu_y + 0.1, qu_z + 0.1, '{}'.format(name))
    ax.view_init(15, 24)

# ----------------------------------------------------#
#   使用范例
# fig = plt.figure()
# ax = Create_sphere(fig)
# Bloch(2,pi,pi/4,ax,'a')     
# Bloch(2,pi/6,pi/4,ax,'b') 
# set_axes_equal(ax)
# plt.show()
# ----------------------------------------------------#




