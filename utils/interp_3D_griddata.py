import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata

def interp3D(arr, domain_size):
    # print(arr.shape)
    points = arr[:, :-1]

    xmin = arr[:, 0].min()
    ymin = arr[:, 1].min()
    zmin = arr[:, 2].min()
    xmax = arr[:, 0].max()
    ymax = arr[:, 1].max()
    zmax = arr[:, 2].max()

    X = np.linspace(xmin, xmax, domain_size[0])
    Y = np.linspace(ymin, ymax, domain_size[1])
    Z = np.linspace(zmin, zmax, domain_size[2])

    x, y, z = np.meshgrid(X, Y, Z)

    x = x.reshape(domain_size[0] * domain_size[1] * domain_size[2], 1)
    y = y.reshape(domain_size[0] * domain_size[1] * domain_size[2], 1)
    z = z.reshape(domain_size[0] * domain_size[1] * domain_size[2], 1)
    sol = arr[:, -1:]

    iu = griddata(points, sol, (x, y, z), method="linear", fill_value=0.0)

    iu = iu.reshape(domain_size[0], domain_size[1], domain_size[2])

    return iu


nx = 64
ny = 64
nz = 64
box_size = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]  # All dimensions in meters

# Define source term
grid_x, grid_y, grid_z = np.meshgrid(np.linspace(box_size[0][0], box_size[0][1], nx),
                                 np.linspace(box_size[1][0], box_size[1][1], ny),
                                 np.linspace(box_size[2][0], box_size[2][1], nz))

grid_x = np.transpose(grid_x, (1, 0, 2))
grid_y = np.transpose(grid_y, (1, 0, 2))
grid_z = np.transpose(grid_z, (1, 0, 2))
print('grid_x shape: ', grid_x.shape)

coordinates = np.concatenate((grid_x.reshape(-1,1), grid_y.reshape(-1,1), grid_z.reshape(-1,1)), axis = 1)
# print('coordinates shape: ', coordinates.shape)
# raise
num_cases = 1000
condition = []
solution = []
# for i in range(num_cases, num_cases+10):
domain_size = (128, 128, 128)
# domain_size = (8, 8, 8)
for i in range(num_cases):

    sol = np.load(f'./data_multiGS_64_new_1000/solution_{i}.npy')
    source = np.load(f'./data_multiGS_64_new_1000/source_term_{i}.npy')

    sol_reshape = sol.reshape(-1, 1)
    source_reshape = source.reshape(-1, 1)
    sol_arr = np.concatenate((coordinates, sol_reshape), axis=1)
    source_arr = np.concatenate((coordinates, source_reshape), axis=1)

    sol = interp3D(sol_arr, domain_size)
    source = interp3D(source_arr, domain_size)
    print('sol shape: ', sol.shape)
    print('source shape: ', source.shape)

    fig = plt.figure()
    ax = fig.add_subplot(121)
    im1 = ax.imshow(sol[sol.shape[0]//2], cmap='jet')
    ax.set_title('solution')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im1, cax=cax, orientation='vertical')


    ax = fig.add_subplot(122)
    im1 = ax.imshow(source[sol.shape[0]//2], cmap='jet')
    ax.set_title('source')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im1, cax=cax, orientation='vertical')
    # plt.show()
    plt.savefig(f'{i}.jpg')
    plt.close()


    np.save('interp_sol.npy', sol)
    np.save('interp_source.npy', source)

    raise
    solution.append(sol)
    condition.append(source)



    # fig = plt.figure()
    # ax = fig.add_subplot(121)
    # im1 = ax.imshow(sol[32], cmap='jet')
    # ax.set_title('solution')
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes('right', size='5%', pad=0.05)
    # fig.colorbar(im1, cax=cax, orientation='vertical')


    # ax = fig.add_subplot(122)
    # im1 = ax.imshow(source[32], cmap='jet')
    # ax.set_title('source')
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes('right', size='5%', pad=0.05)
    # fig.colorbar(im1, cax=cax, orientation='vertical')
    # plt.show()
    # # plt.savefig(f'{j}.jpg')
    # plt.close()
    # raise
solution = np.asarray(solution)
condition = np.asarray(condition)
np.save('solution.npy', solution)
np.save('condition.npy',condition)
# np.save('solution_test.npy', solution)
# np.save('condition_test.npy',condition)
print('solution: ', solution.shape)
print('condition: ', condition.shape)
raise
