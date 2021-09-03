import os
import glob
import numpy as np
import pickle
import matplotlib.pyplot as plt

cur_path = os.getcwd()
data_dir = cur_path + '\\Fluent_outputs\\ML_Solver_data_96\\block_data'
data_files = glob.glob(data_dir + '\\*.pickle')

all_case_dir = cur_path + '\\Fluent_outputs\\Fluent_outputs_v04_96\\'
case_files = os.listdir(all_case_dir)



# cur_path = os.getcwd()
# data_dir = cur_path + '\\Fluent_outputs\\ML_Solver_data_np_cell\\block_data'
# data_files = glob.glob(data_dir + '\\*.pickle')
#
# all_case_dir = cur_path + '\\Fluent_outputs\\Fluent_outputs_v02_382_np\\'
# case_files = os.listdir(all_case_dir)


# power_maps = np.empty((0, 200, 200))
# temp_mins = np.empty((0, 1))
# params = np.empty((0, 2))
# cnt = 0
# for case_file, data_file in zip(case_files, data_files):
#     if case_file == data_file.split('\\')[-1].split('.')[0][13:]:
#         print('cnt=', cnt)
#         cnt += 1
#         power_map = np.load(glob.glob(all_case_dir+case_file + '\\' + "*.npy")[0])
#         power_maps = np.concatenate((power_maps, power_map.reshape((1, 200, 200))), axis=0)
#
#         with open(data_file, "rb") as fp:  # Unpickling
#             data = pickle.load(fp)
#         temp_min = np.asarray([[data['solution']['temperature'][0].min()]])
#         temp_mins = np.concatenate((temp_mins, temp_min), axis=0)
#
#         param_str = case_file.split('_')
#         AmbT = int(param_str[2])
#         FinCount = int(param_str[4])
#         if param_str[6] == 'Off':
#             FinCount = 0
#         Total_power = int(param_str[-1])
#
#         param = np.asarray([[AmbT, FinCount]])
#         params = np.concatenate((params, param), axis=0)
#
# T_min_predictor_data_dir = cur_path + '\\T_min_predictor\\data\\'
# os.makedirs(T_min_predictor_data_dir, mode=0o777,exist_ok=True)
# np.savez_compressed(T_min_predictor_data_dir + 'power_maps_362.npz', power_maps)
# np.savez_compressed(T_min_predictor_data_dir + 'temp_mins_362.npz', temp_mins)
# np.savez_compressed(T_min_predictor_data_dir + 'params_362.npz', params)

# fig = plt.figure()
# ax = fig.add_subplot(111)
# true = temp_mins
# x = np.arange(true.shape[0])
# plt.scatter(x, true, c = 'b')
# ax.legend(['Min temp'])
# ax.set_xlabel('case number')
# ax.set_ylabel('Temperature')
# T_min_predictor_folder = '.\\T_min_predictor\\'
# plt.savefig(T_min_predictor_folder + 'Min_pred_all_362')

T_min_predictor_folder = '.\\T_min_predictor\\'
os.makedirs(T_min_predictor_folder, mode=0o777, exist_ok=True)
train_data_folder = '.\\T_min_predictor\\data\\'
params = np.load(train_data_folder + 'params_362.npz')['arr_0']
power_maps = np.load(train_data_folder + 'power_maps_362.npz')['arr_0']
temp_mins = np.load(train_data_folder + 'temp_mins_362.npz')['arr_0']

params[:, 0] = (params[:, 0] - params[:, 0].min()) / (params[:, 0].max() - params[:, 0].min())
params[:, 1] = (params[:, 1] - params[:, 1].min()) / (params[:, 1].max() - params[:, 1].min())
power_maps = (power_maps - power_maps.min()) / (power_maps.max() - power_maps.min())
# temp_mins /= 273.15
# temp_mins -= 1

# temp_mins -= 273.15
lower_bound = 273.15
upper_bound = 335
# temp_mins = (temp_mins - temp_mins.min()) / (temp_mins.max() - temp_mins.min())
temp_mins = (temp_mins - lower_bound) / (upper_bound - lower_bound)


import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, BatchNormalization, Flatten, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import relu

# input_power_map = Input(shape=(200, 200, 1))
# input_param = Input(shape=(2,))

class net(Model):
    def __init__(self, pw_map_size, param_size, filters, hidden, out_size):
        super(net, self).__init__()
        self.pw_map_size = pw_map_size
        self.param_size = param_size
        self.hidden = hidden
        self.filters = filters
        self.out_size = out_size

        self.map_model = self.map_net()
        self.param_model = self.param_net()
        self.min_temp_model = self.min_temp_net()

    def conv_Block(self, x, BN, filters):
        for i in range(2):
            x = Conv2D(filters, 3, padding='same')(x)
            if BN:
                x = BatchNormalization()(x)
            x = relu(x)
        return x

    def map_net(self):
        BN = False
        inputs = Input(shape=(self.pw_map_size,self.pw_map_size,1))
        x = inputs
        for i in range(4):
            x = self.conv_Block(x, BN, self.filters)
            x = MaxPooling2D(pool_size=2, strides=None, padding='same')(x)
        x = Conv2D(1, 3, padding='same', activation='relu')(x)
        map_model = Model(inputs=[inputs], outputs=[x])
        return map_model

    def param_net(self):
        BN = False
        inputs = Input(self.param_size, )
        x = inputs
        for i in range(2):
            x = Dense(self.hidden, activation=relu)(x)
        # x = Dense(self.out_size, activation=relu)(x)
        param_model = Model(inputs=[inputs], outputs=[x])
        return param_model

    def min_temp_net(self):
        input_map = self.map_model.inputs
        input_param = self.param_model.inputs
        out_map = self.map_model.outputs
        out_map = Flatten()(out_map[0])
        out_map = Dense(self.hidden, activation='relu')(out_map)

        out_param = self.param_model.outputs[0]

        merge = Concatenate()([out_map, out_param])
        merge = Dense(self.hidden, activation='relu')(merge)
        merge = Dense(self.out_size)(merge)
        min_temp_model = Model(inputs=[input_map, input_param], outputs=[merge])
        return min_temp_model

    # def call(self, x_map, x_param):
    #     out = self.min_temp_model([x_map, x_param])
    #     return out

    # def call(self, X):
    #     out = self.min_temp_model(X)
    #     return out

x_map = np.expand_dims(power_maps, -1)
x_param = params
y_temp_min = temp_mins

print("x_map shape: ", x_map.shape)
print("x_param shape: ", x_param.shape)
print('y_temp_min shape: ', y_temp_min.shape)
# raise
num_filter_hidden = 1
pw_map_size = 200
param_size = 2
filters = 8
hidden = 8
out_size = 1
model = net(pw_map_size, param_size, filters, hidden, out_size).min_temp_model

ckpt_dir = T_min_predictor_folder + 'checkpoint\\'
os.makedirs(ckpt_dir, mode=0o777, exist_ok=True)
work_dir = T_min_predictor_folder
# my_callbacks = [
# #     tf.keras.callbacks.EarlyStopping(patience=2),
#     tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_dir + 'model.{epoch:02d}-{val_loss:.2f}.h5'),
#     tf.keras.callbacks.TensorBoard(log_dir = work_dir + '\\logs'),
# ]

def scheduler(epoch, lr):
  if epoch < 100:
    return lr
  else:
    return lr * tf.math.exp(-0.1)
lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

my_callbacks = [
#     tf.keras.callbacks.EarlyStopping(patience=2),
#     tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_dir + 'model.{epoch:02d}-{val_loss:.2f}.h5',save_freq='epoch'),\
    tf.keras.callbacks.ModelCheckpoint(ckpt_dir + 'model.h5',
        monitor='val_loss', save_weights_only=False, save_best_only=True, period=50, save_freq='epoch'),
    tf.keras.callbacks.TensorBoard(log_dir = work_dir + '\\logs'),
    # lr_callback
]

lr = 1e-3
optimizer = Adam(lr = lr)
model.compile(optimizer=optimizer, loss='mse')
history = model.fit(
    [x_map, x_param], y_temp_min,
    epochs = 5000,
    batch_size=8,
    shuffle=True,
    validation_split=0.1,
    callbacks=my_callbacks
)

train_loss = history.history['loss']
val_loss = history.history['val_loss']
fig = plt.figure()
ax = fig.add_subplot(111)
ax.semilogy(train_loss)
ax.semilogy(val_loss)
ax.legend(['train', 'validation'])
plt.savefig(T_min_predictor_folder + 'loss_history')

idx_0 = params[:,0] == 0
idx_30 = params[:,0] == 0.5
idx_60 = params[:,0] == 1

model.load_weights(ckpt_dir + 'model.h5')



fig = plt.figure()
ax = fig.add_subplot(111)
pred = model.predict([x_map, x_param]) * (upper_bound - lower_bound) + lower_bound
true = y_temp_min * (upper_bound - lower_bound) + lower_bound
x = np.arange(true.shape[0])
plt.scatter(x, pred, c = 'r')
plt.scatter(x, true, c = 'b')
ax.legend(['Prediction', 'Truth'])
plt.savefig(T_min_predictor_folder + 'Min_pred_all')

fig = plt.figure()
ax = fig.add_subplot(111)
pred = (model.predict([x_map, x_param]) * (upper_bound - lower_bound) + lower_bound)[idx_0]
true = (y_temp_min * (upper_bound - lower_bound) + lower_bound)[idx_0]
x = np.arange(true.shape[0])
plt.scatter(x, pred, c = 'r')
plt.scatter(x, true, c = 'b')
ax.legend(['Prediction', 'Truth'])
ax.set_title(f'Ambient = {273.15 + 0}')
plt.savefig(T_min_predictor_folder + 'Min_pred_Ambient_0')

fig = plt.figure()
ax = fig.add_subplot(111)
pred = (model.predict([x_map, x_param]) * (upper_bound - lower_bound) + lower_bound)[idx_30]
true = (y_temp_min * (upper_bound - lower_bound) + lower_bound)[idx_30]
x = np.arange(true.shape[0])
plt.scatter(x, pred, c = 'r')
plt.scatter(x, true, c = 'b')
ax.legend(['Prediction', 'Truth'])
ax.set_title(f'Ambient = {273.15 + 30}')
plt.savefig(T_min_predictor_folder + 'Min_pred_Ambient_30')

fig = plt.figure()
ax = fig.add_subplot(111)
pred = (model.predict([x_map, x_param]) * (upper_bound - lower_bound) + lower_bound)[idx_60]
true = (y_temp_min * (upper_bound - lower_bound) + lower_bound)[idx_60]
x = np.arange(true.shape[0])
plt.scatter(x, pred, c = 'r')
plt.scatter(x, true, c = 'b')
ax.legend(['Prediction', 'Truth'])
ax.set_title(f'Ambient = {273.15 + 60}')
plt.savefig(T_min_predictor_folder + 'Min_pred_Ambient_60')

raise
# for data_file in data_files:
