import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm

def interp2D(arr, domain_size, interp_domain_size):

    xmin = 0.0
    ymin = 0.0

    xmax = 1.0
    ymax = 1.0


    X = np.linspace(xmin, xmax, domain_size[0])
    Y = np.linspace(ymin, ymax, domain_size[1])



    sol = arr

    iu = RegularGridInterpolator((X, Y), sol, method="linear", fill_value=0.0)

    X_interp = np.linspace(xmin, xmax, interp_domain_size[0])
    Y_interp = np.linspace(ymin, ymax, interp_domain_size[1])

    X_mg, Y_mg = np.meshgrid(X_interp, Y_interp)
    X_mg = np.transpose(X_mg, (1, 0))
    Y_mg = np.transpose(Y_mg, (1, 0))

    X_mg, Y_mg = X_mg.reshape(-1, 1), Y_mg.reshape(-1, 1)
    pts = np.concatenate((X_mg, Y_mg), axis=-1)
    sol_interp = iu(pts).reshape(interp_domain_size[0], interp_domain_size[1])

    # fig = plt.figure()
    # ax = fig.add_subplot(121)
    # im1 = ax.imshow(arr, cmap='jet')
    # ax.set_title('solution')
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes('right', size='5%', pad=0.05)
    # fig.colorbar(im1, cax=cax, orientation='vertical')


    # ax = fig.add_subplot(122)
    # im1 = ax.imshow(sol_interp, cmap='jet')
    # ax.set_title('interp solution')
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes('right', size='5%', pad=0.05)
    # fig.colorbar(im1, cax=cax, orientation='vertical')
    # plt.show()
    # # plt.savefig(f'{i}.jpg')
    # plt.close()

    return sol_interp




folders = ['test', 'train']
domain_size = (128, 128)
interp_domain_size = (1024, 1024)
for folder in folders:
    sol_train = np.load(f'./{folder}/solution.npy').astype(np.float32)
    con_train = np.load(f'./{folder}/condition.npy').astype(np.float32)
    print('sol_train shape: ', sol_train.shape)
    raise
    condition = []
    solution = []
    for (sol, source) in tqdm(zip(sol_train, con_train)):

        sol_arr = sol
        source_arr = source


        sol = interp2D(sol_arr, domain_size, interp_domain_size)
        source = interp2D(source_arr, domain_size, interp_domain_size)
        # print('sol shape: ', sol.shape)
        # print('source shape: ', source.shape)

        # fig = plt.figure()
        # ax = fig.add_subplot(121)
        # im1 = ax.imshow(sol, cmap='jet')
        # ax.set_title('solution')
        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes('right', size='5%', pad=0.05)
        # fig.colorbar(im1, cax=cax, orientation='vertical')


        # ax = fig.add_subplot(122)
        # im1 = ax.imshow(source, cmap='jet')
        # ax.set_title('source')
        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes('right', size='5%', pad=0.05)
        # fig.colorbar(im1, cax=cax, orientation='vertical')
        # plt.show()
        # plt.close()
        # raise


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
    np.save(f'./interp/{folder}/solution.npy', solution)
    np.save(f'./interp/{folder}/condition.npy',condition)
    # np.save('solution_test.npy', solution)
    # np.save('condition_test.npy',condition)
    print('solution: ', solution.shape)
    print('condition: ', condition.shape)
