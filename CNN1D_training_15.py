"""# CNN-1D-15分法-訓練-tensorflow2.0"""

import tensorflow as tf
import numpy as np


def loading_data():
  signal_data = np.load('15_class_signal_train_256_1D_for_all_data_high_8_2.npy')
  symbol_data = np.load('15_class_symbol_train_256_for_all_data_high_8_2.npy')
  signal_test = np.load('15_class_signal_test_256_1D_for_all_data_high_8_2.npy')
  symbol_test = np.load('15_class_symbol_test_256_for_all_data_high_8_2.npy')

  return signal_data, symbol_data, signal_test, symbol_test

def random_process_data(imgs, labels):
  # 打散數據集
  indices = np.random.permutation(imgs.shape[0])
  imgs = imgs[indices]  
  labels = labels[indices]

  return imgs, labels

def build_model():
  model = tf.keras.models.Sequential([
  tf.keras.layers.Conv1D(4, 21, strides= 1,padding='same', activation= 'relu'),
  tf.keras.layers.MaxPooling1D(pool_size=3, strides=2, padding='same'),
  tf.keras.layers.Conv1D(16, 23, strides=1, padding='same', activation='relu'),
  tf.keras.layers.MaxPooling1D(pool_size=3, strides=2, padding='same'),
  tf.keras.layers.Conv1D(32, 25, strides=1, padding='same', activation='relu'),
  tf.keras.layers.AveragePooling1D(pool_size=3, strides=2, padding='same'),
  tf.keras.layers.Conv1D(64, 27, strides=1, padding= 'same', activation='relu'),
  tf.keras.layers.MaxPooling1D(pool_size=3, strides=2, padding='same'),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Dense(15, activation='softmax')
  ])
  model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
  # model.summary() # 秀出模型架構
  return model

def label_to_number(label):
  for i in range(len(label)):
    if label[i] == 'N':
      label[i] = 0
    elif label[i] == 'e':
      label[i] = 1
    elif label[i] == 'j':
      label[i] = 2
    elif label[i] == 'L':
      label[i] = 3
    elif label[i] == 'R':
      label[i] = 4
    elif label[i] == 'A':
      label[i] = 5
    elif label[i] == 'a':
      label[i] = 6
    elif label[i] == 'J':
      label[i] = 7
    elif label[i] == 'S':
      label[i] = 8
    elif label[i] == 'V':
      label[i] = 9
    elif label[i] == 'E':
      label[i] = 10
    elif label[i] == 'F':
      label[i] = 11
    elif label[i] == '/':
      label[i] = 12
    elif label[i] == 'f':
      label[i] = 13
    elif label[i] == 'Q':
      label[i] = 14
    # else:
    #   print(label[i])
  label = label.astype(int)
  return label


signal_data, symbol_data, signal_test, symbol_test = loading_data()

signal_data, symbol_data = random_process_data(signal_data, symbol_data)

signal_data = signal_data.reshape(signal_data.shape[0], signal_data.shape[1], 1) #重新組成->(trainX資料長度,step=300or500,hdden=1)
signal_test = signal_test.reshape(signal_test.shape[0], signal_test.shape[1], 1)

symbol_data = label_to_number(symbol_data)
symbol_test = label_to_number(symbol_test)

save_file_path = './model_15.h5'
model = build_model()
# model = tf.keras.models.load_model(save_file_path)

#若驗證準確率筆上次訓練來得高>>>>存檔
checkpoint = tf.keras.callbacks.ModelCheckpoint(save_file_path, monitor='val_acc', verbose=1, save_best_only=True,mode='max')
callbacks_list = [checkpoint]

# model.build(signal_data.shape) 
# model.summary()
model.fit(signal_data, symbol_data, epochs=50, batch_size=128, validation_split=0.3, callbacks=callbacks_list)
model.save(save_file_path)
print("model已儲存")

test_loss, test_acc = model.evaluate(signal_test, symbol_test, verbose=2)
print("Test_loss: ", test_loss)
print("Test_acc: ", test_acc)


