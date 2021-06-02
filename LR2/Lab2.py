import pandas as pd
import datetime
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler()

# Загружаем данные и удаляем неиспользуемые поля (в данном случае Timestamp)
csv_path = r"C:\Users\Andrey\Downloads\btc\btc"
df = pd.read_csv(csv_path)
date_time = pd.to_datetime(df.pop('Timestamp'), format='%Y-%m-%d %H:%M:%S')
timestamp_s = date_time.map(datetime.datetime.timestamp)

# Отображение исходных данных
plot_cols = ['market-price']
plot_features = df[plot_cols]
plot_features.index = date_time
_ = plot_features.plot(subplots=True)

# Разделить на обучение и тестирование
# По методичке мы должны разбивать свои данные на тренировачные, валидные,
# тестовые (70%, 20%, 10%). Мы выбрали тренировочный набор на 30 дней,
# что означает, что мы собираемся протестировать нашу модель за последний месяц.

prediction_days = 30

train_df = df[:len(df)-prediction_days]
test_df = df[len(df)-prediction_days:]
val_df = test_df

# Размер выходного состояния, используемых для ячейки LSTM.
num_units = 4
# Функция активации используется для ячейки LSTM
activation_function = 'sigmoid'

lstm_model = tf.keras.models.Sequential([
    # Shape [batch, time, features] => [batch, time, lstm_units]
    tf.keras.layers.LSTM(units=num_units, activation=activation_function,
                         input_shape=(None, 1)),
    # Shape => [batch, time, features]
    tf.keras.layers.Dense(units=1)
])

# нейронные сети обычно обучаются партиями, что означает, что на каждой
# итерации мы выбираем 5 примеров из нашего обучающего набора и используем
# их для обучения.
batch_size = 5
num_epochs = 100

training_set = train_df.values
training_set = min_max_scaler.fit_transform(training_set)

x_train = training_set[0:len(training_set)-1]
y_train = training_set[1:len(training_set)]
x_train = np.reshape(x_train, (len(x_train), 1, 1))

lstm_model.compile(optimizer=tf.optimizers.Adam(),
              loss=tf.losses.MeanSquaredError())
lstm_model.fit(x_train, y_train, batch_size = batch_size, epochs = num_epochs)

# Сохранение модели в файл
output = 'model'
tf.keras.models.save_model(lstm_model, output)

test_set = test_df.values
inputs = np.reshape(test_set, (len(test_set), 1))
inputs = min_max_scaler.transform(inputs)
inputs = np.reshape(inputs, (len(inputs), 1, 1))

predicted_price = lstm_model.predict(inputs)
predicted_price = min_max_scaler.inverse_transform(predicted_price)

plt.figure(figsize=(25, 25), dpi=80, facecolor = 'w', edgecolor = 'k')
plt.plot(test_set[:, 0], color='red', label='Настоящая цена биткоина')
plt.plot(predicted_price[:, 0], color = 'blue', label = 'Прогнозированная цена биткоина')
plt.title('Прогноз биткоина', fontsize = 40)
plt.xlabel('Время', fontsize=40)
plt.ylabel('Биткоин - USD', fontsize = 40)
plt.legend(loc = 'best')
plt.show()