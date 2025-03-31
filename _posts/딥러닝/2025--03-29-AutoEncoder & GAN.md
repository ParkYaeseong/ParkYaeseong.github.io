---
title: 딥러닝을 이용한 비지도 학습: 오토인코더와 GAN
date: 2025-03-31 05:11:00 +09:00
categories: [딥러닝]
tags: [딥러닝, AutoEncoder, GAN]
---

##   딥러닝을 이용한 비지도 학습: 오토인코더와 GAN

 ###   서론

 이 블로그 게시글에서는 딥러닝을 사용한 비지도 학습 기법인 오토인코더와 생성적 적대 신경망(GAN)을 살펴봅니다. 비지도 학습은 레이블이 없는 데이터를 다루며, 숨겨진 패턴이나 구조를 발견하는 것을 목표로 합니다. 오토인코더는 차원 축소 및 노이즈 제거와 같은 작업에 사용되는 반면, GAN은 훈련 데이터와 유사한 새로운 데이터를 생성하는 데 강력합니다.

 ###   오토인코더

 ####   오토인코더란 무엇인가?

 오토인코더는 입력값을 출력값으로 복사하도록 학습하는 신경망의 한 유형입니다. 크게 두 부분으로 구성됩니다.

 * **인코더**: 입력을 저차원 표현(잠재 공간)으로 압축합니다.
 * **디코더**: 잠재 공간 표현에서 원본 입력을 재구성합니다.

 ####   기본 오토인코더 예제


 import tensorflow as tf
 from tensorflow.keras import layers, models
 import numpy as np
 

 # 샘플 입력 데이터
 input_data = np.random.rand(1, 4, 4, 3).astype(np.float32)
 print("입력 데이터 형태:", input_data.shape)
 

 # 간단한 오토인코더 모델 정의
 model = models.Sequential([
  layers.Conv2DTranspose(5, kernel_size=3, strides=2,
  activation='relu', padding='same', input_shape=(4, 4, 3)),
 ])
 

 # 레이어 및 출력 형태를 보여주는 모델 요약
 model.summary()


 ####   MNIST를 위한 컨볼루션 오토인코더

 이 예제는 이미지 재구성에 초점을 맞춘 MNIST 데이터세트에 대한 컨볼루션 오토인코더를 보여줍니다.


 import tensorflow as tf
 from tensorflow.keras.datasets import mnist
 from tensorflow.keras.models import Sequential
 from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
 import matplotlib.pyplot as plt
 import numpy as np
 

 # MNIST 데이터세트 로드 및 전처리
 (X_train, _), (X_test, _) = mnist.load_data()
 X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32') / 255
 X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32') / 255
 

 # 오토인코더 모델 구축
 autoencoder = Sequential()
 

 # --- 인코더 ---
 autoencoder.add(Conv2D(16, kernel_size=3, padding='same',
  input_shape=(28, 28, 1), activation='relu'))
 autoencoder.add(MaxPooling2D(pool_size=2, padding='same'))
 autoencoder.add(Conv2D(8, kernel_size=3, padding='same', activation='relu'))
 autoencoder.add(MaxPooling2D(pool_size=2, padding='same'))
 autoencoder.add(Conv2D(8, kernel_size=3, strides=2,
  padding='same', activation='relu'))
 

 # --- 디코더 ---
 autoencoder.add(Conv2D(8, kernel_size=3, padding='same', activation='relu'))
 autoencoder.add(UpSampling2D())
 autoencoder.add(Conv2D(8, kernel_size=3, padding='same', activation='relu'))
 autoencoder.add(UpSampling2D())
 autoencoder.add(Conv2D(16, kernel_size=3, padding='same',
  activation='relu'))
 autoencoder.add(Conv2D(1, kernel_size=5, padding='valid',
  activation='sigmoid'))  # 출력 레이어
 

 autoencoder.summary()
 

 # 오토인코더 컴파일 및 훈련
 autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
 autoencoder.fit(X_train, X_train, epochs=50, batch_size=128,
  validation_data=(X_test, X_test))
 

 # 원본 및 재구성된 이미지 시각화
 random_test = np.random.randint(X_test.shape[0], size=5)
 ae_imgs = autoencoder.predict(X_test)
 

 plt.figure(figsize=(7, 2))
 for i, image_idx in enumerate(random_test):
  ax = plt.subplot(2, 7, i + 1)
  plt.imshow(X_test[image_idx].reshape(28, 28))  # 원본
  ax.axis('off')
  ax = plt.subplot(2, 7, 7 + i + 1)
  plt.imshow(ae_imgs[image_idx].reshape(28, 28))  # 재구성됨
  ax.axis('off')
 plt.show()

 ####   손실 함수: 이진 교차 엔트로피

 import tensorflow as tf
 

 # 이진 교차 엔트로피 손실 예제
 x = tf.random.uniform((1, 28, 28, 1), minval=0., maxval=1.)
 x_pred = tf.random.uniform((1, 28, 28, 1), minval=0., maxval=1.)
 bce = tf.keras.losses.BinaryCrossentropy()
 loss = bce(x, x_pred)
 print("BCE 손실:", loss.numpy())


 ####   특징 추출을 위한 오토인코더

 오토인코더는 입력 데이터의 차원을 줄이는 특징 추출에도 사용할 수 있습니다.

 import tensorflow as tf
 from tensorflow.keras import layers, losses
 from tensorflow.keras.datasets import fashion_mnist
 from tensorflow.keras.models import Model
 import matplotlib.pyplot as plt
 

 # 패션 MNIST 데이터세트 로드 및 전처리
 (X_train, _), (X_test, _) = fashion_mnist.load_data()
 X_train = X_train.astype('float32') / 255.
 X_test = X_test.astype('float32') / 255.
 print(X_train.shape)
 print(X_test.shape)
 

 # 특징 추출을 위한 오토인코더 모델 정의
 latent_dim = 64  # 잠재 공간의 크기
 

 class Autoencoder(Model):
  def __init__(self, latent_dim):
  super(Autoencoder, self).__init__()
  self.latent_dim = latent_dim
  self.encoder = tf.keras.Sequential([
  layers.Flatten(),
  layers.Dense(latent_dim, activation='relu'),
  ])
  self.decoder = tf.keras.Sequential([
  layers.Dense(784, activation='sigmoid'),
  layers.Reshape((28, 28))
  ])
 

  def call(self, x):
  encoded = self.encoder(x)
  decoded = self.decoder(encoded)
  return decoded
 

 autoencoder = Autoencoder(latent_dim)
 autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
 

 # 오토인코더 훈련
 autoencoder.fit(X_train, X_train, epochs=10, shuffle=True,
  validation_data=(X_test, X_test))
 

 # 특징 추출을 위해 인코더 사용
 encoded_imgs = autoencoder.encoder(X_test).numpy()
 print("인코딩된 이미지 형태:", encoded_imgs.shape)  # (10000, 64)
 

 # 원본 및 재구성된 이미지 시각화
 decoded_imgs = autoencoder(X_test).numpy()
 

 n = 10
 plt.figure(figsize=(20, 4))
 for i in range(n):
  ax = plt.subplot(2, n, i + 1)
  plt.imshow(X_test[i])
  plt.title("원본")
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
 

  ax = plt.subplot(2, n, i + 1 + n)
  plt.imshow(decoded_imgs[i])
  plt.title("재구성됨")
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
 plt.show()

 ####   노이즈 제거 오토인코더

 오토인코더는 이미지에서 노이즈를 제거하도록 훈련할 수도 있습니다.

 import tensorflow as tf
 from tensorflow.keras import layers, losses
 from tensorflow.keras.datasets import fashion_mnist
 from tensorflow.keras.models import Model
 import matplotlib.pyplot as plt
 

 # 패션 MNIST 데이터세트 로드 및 전처리
 (x_train, _), (x_test, _) = fashion_mnist.load_data()
 x_train = x_train.astype('float32') / 255.
 x_test = x_test.astype('float32') / 255.
 x_train = x_train[..., tf.newaxis]
 x_test = x_test[..., tf.newaxis]
 print(x_train.shape)
 

 # 이미지에 노이즈 추가
 noise_factor = 0.2
 x_train_noisy = x_train + noise_factor * tf.random.normal(
  shape=x_train.shape)
 x_test_noisy = x_test + noise_factor * tf.random.normal(shape=x_test.shape)
 x_train_noisy = tf.clip_by_value(x_train_noisy, clip_value_min=0.,
  clip_value_max=1.)
 x_test_noisy = tf.clip_by_value(x_test_noisy, clip_value_min=0.,
  clip_value_max=1.)
 

 # 노이즈가 추가된 이미지 시각화
 n = 10
 plt.figure(figsize=(20, 2))
 for i in range(n):
  ax = plt.subplot(1, n, i + 1)
  plt.title("원본 + 노이즈")
  plt.imshow(tf.squeeze(x_test_noisy[i]))
  plt.gray()
 plt.show()
 

 # 노이즈 제거 오토인코더 모델 정의
 class Denoise(Model):
  def __init__(self):
  super(Denoise, self).__init__()
  self.encoder = tf.keras.Sequential([
  layers.Input(shape=(28, 28, 1)),
  layers.Conv2D(16, (3, 3), activation='relu', padding='same',
  strides=2),  # 14x14x16
  layers.Conv2D(8, (3, 3), activation='relu', padding='same',
  strides=2)  # 7x7x8
  ])
  self.decoder = tf.keras.Sequential([
  layers.Conv2DTranspose(8, kernel_size=3, strides=2,
  activation='relu', padding='same'),  # 14x14x8
  layers.Conv2DTranspose(16, kernel_size=3, strides=2,
  activation='relu', padding='same'),  # 28x28x16
  layers.Conv2DTranspose(1, kernel_size=(3, 3), activation='relu',
  padding='same')  # 28x28x1
  ])
 

  def call(self, x):
  encoded = self.encoder(x)
  decoded = self.decoder(encoded)
  return decoded
 

 autoencoder = Denoise()
 autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
 

 # 노이즈 제거 오토인코더 훈련
 autoencoder.fit(x_train_noisy, x_train, epochs=10, shuffle=True,
  validation_data=(x_test_noisy, x_test))
 

 # 노이즈가 있는 이미지와 노이즈가 제거된 이미지 시각화
 n = 10
 plt.figure(figsize=(20, 2))
 for i in range(n):
  ax = plt.subplot(2, n, i + 1)
  plt.title("원본 + 노이즈")
  plt.imshow(tf.squeeze(x_test_noisy[i]))  # squeeze : 1
  plt.title("노이즈 이미지")
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
 

  ax = plt.subplot(2, n, i + 1 + n)
  plt.title("원본 + 노이즈")
  plt.imshow(tf.squeeze(decoded_imgs[i]))  # squeeze : 1
  plt.title("재구성된 이미지")
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
 plt.show()

 ####   이상 감지

 오토인코더는 정상 데이터로 훈련한 다음 재구성 오류가 큰 데이터 포인트를 이상으로 식별하여 이상 감지에 사용할 수 있습니다.

 import numpy as np
 import matplotlib.pyplot as plt
 import tensorflow as tf
 from tensorflow.keras import layers, models
 

 # 패션 MNIST 데이터세트 로드 및 전처리
 (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
 x_train = x_train.astype('float32') / 255.
 x_test = x_test.astype('float32') / 255.
 x_train = np.reshape(x_train, (-1, 28, 28, 1))
 x_test = np.reshape(x_test, (-1, 28, 28, 1))
 

 # 정상 클래스 정의 및 정상/이상 샘플 추출
 NORMAL_CLASS = 0
 x_train_normal = x_train[y_train == NORMAL_CLASS]
 x_test_normal = x_test[y_test == NORMAL_CLASS]
 x_test_anomaly = x_test[y_test != NORMAL_CLASS]
 print(f"정상 샘플: {len(x_train_normal)}")
 print(f"정상 테스트 샘플: {len(x_test_normal)}",
  f"이상 테스트 샘플: {len(x_test_anomaly)}")
 

 # 오토인코더 모델 구축
 def build_autoencoder():
  input_img = layers.Input(shape=(28, 28, 1))
  x = layers.Conv2D(32, 3, activation='relu', padding='same')(input_img)
  x = layers.MaxPooling2D(2, padding='same')(x)
  x = layers.Conv2D(16, 3, activation='relu', padding='same')(x)
  encoded = layers.MaxPooling2D(2, padding='same')(x)
  x = layers.Conv2D(16, 3, activation='relu', padding='same')(encoded)
  x = layers.UpSampling2D(2)(x)
  x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
  x = layers.UpSampling2D(2)(x)
  decoded = layers.Conv2D(1, 3, activation='sigmoid', padding='same')(x)
  autoencoder = models.Model(input_img, decoded)
  autoencoder.compile(optimizer='adam', loss='mse')
  return autoencoder
 

 autoencoder = build_autoencoder()
 autoencoder.summary()
 

 # 정상 데이터에 대해 오토인코더 훈련
 history = autoencoder.fit(x_train_normal, x_train_normal,
  epochs=10,
  batch_size=126,
  shuffle=True,
  validation_data=(x_test_normal, x_test_normal))
 

 # 재구성 오류를 계산하는 함수
 def compute_reconstruction_errors(model, data):
  reconstructed = model.predict(data)
  errors = np.mean(np.square(data - reconstructed), axis=(1, 2, 3))
  return errors, reconstructed
 

 # 이상 감지 임계값 설정
 normal_errors, _ = compute_reconstruction_errors(autoencoder,
  x_test_normal)
 threshold = np.percentile(normal_errors, 95)
 

 # 테스트 데이터에서 이상 감지
 anomaly_errors, anomaly_recon = compute_reconstruction_errors(autoencoder,
  x_test_anomaly)
 anomaly_detected = anomaly_errors > threshold
 anomaly_rate = np.mean(anomaly_detected)
 

 # 이상 감지 결과 출력
 print(f"\n이상 탐지 경계값: {threshold:.4f}")
 print(
  f"이상값 {np.sum(anomaly_detected)} 전체 데이터:{len(anomaly_detected)}({anomaly_rate * 100:.2f}%)")

 ###   생성적 적대 신경망(GAN)

 ####   GAN이란 무엇인가?

 GAN은 2014년 Ian Goodfellow와 그의 동료들이 설계한 머신러닝 프레임워크의 한 클래스입니다. 두 개의 신경망이 제로섬 게임에서 서로 경쟁하며, 여기서 한 네트워크의 이득은 다른 네트워크의 손실입니다.

 * **생성기**: 새로운 데이터 인스턴스를 생성합니다.
 * **판별기**: 데이터 인스턴스의 진위 여부를 평가합니다(실제 대 가짜).

 ####   KL 발산

 GAN은 확률 분포 간의 차이를 측정하는 것을 포함합니다. 쿨백-라이블러(KL) 발산은 한 확률 분포가 두 번째 예상 확률 분포와 얼마나 다른지를 측정한 값입니다.

 import numpy as np
 

 # KL 발산 계산 예제
 P = np.array([0.3, 0.4, 0.3])
 Q = np.array([0.2, 0.3, 0.5])
 

 kl_divergence = np.sum(P * np.log(P / Q))
 print("KL 발산:", kl_divergence)
 

 P = np.array([0.3, 0.4, 0.3])
 Q = np.array([0.09, 0.99, 0.89])
 

 kl_divergence = np.sum(P * np.log(P / Q))
 print("KL 발산:", kl_divergence)

 ####   MNIST 생성을 위한 GAN

 이 예제는 MNIST와 유사한 이미지를 생성하는 간단한 GAN을 보여줍니다.

 import os
 import numpy as np
 import tensorflow as tf
 from tensorflow.keras.layers import Reshape, Flatten, Dropout, LeakyReLU
 from tensorflow.keras.layers import BatchNormalization, Activation,
  UpSampling2D, Conv2D, Dense, Input
 from tensorflow.keras.models import Sequential, Model
 from tensorflow.keras.datasets import mnist
 import matplotlib.pyplot as plt
 

 # 생성된 이미지를 저장할 디렉토리 생성
 os.makedirs('./gan_images', exist_ok=True)
 

 # 재현성을 위해 랜덤 시드 설정
 np.random.seed(3)
 tf.random.set_seed(3)
 

 # 생성기 모델
 generator = Sequential()
 generator.add(Dense(128 * 7 * 7, input_dim=100,
  activation=LeakyReLU(0.2)))
 generator.add(BatchNormalization())
 generator.add(Reshape((7, 7, 128)))
 generator.add(UpSampling2D())  # 14x14x128
 generator.add(Conv2D(64, kernel_size=5, padding='same'))  # 14x14x64
 generator.add(BatchNormalization())
 generator.add(Activation(LeakyReLU(0.2)))
 generator.add(UpSampling2D())  # 28x28x64
 generator.add(Conv2D(1, kernel_size=5, padding='same',
  activation='tanh'))  # 28x28x1
 generator.summary()
 

 # 판별기 모델
 discriminator = Sequential()
 discriminator.add(Conv2D(64, kernel_size=5, strides=2,
  input_shape=(28, 28, 1), padding='same'))  # 14x14x64
 discriminator.add(Activation(LeakyReLU(0.2)))
 discriminator.add(Dropout(0.3))
 discriminator.add(Conv2D(128, kernel_size=5, strides=2,
  padding='same'))  # 7x7x128
 discriminator.add(Activation(LeakyReLU(0.2)))
 discriminator.add(Dropout(0.3))
 discriminator.add(Flatten())  # 6272
 discriminator.add(Dense(1, activation='sigmoid'))  # 1의 값이 분포차를 계
 discriminator.compile(loss='binary_crossentropy', optimizer='adam')
 discriminator.trainable = False  # 학습하지 않음
 discriminator.summary()
 

 # GAN 모델
 ginput = Input(shape=(100,))  # 노이즈 100개
 dis_output = discriminator(generator(ginput))  # 가짜 이미지를 생성하고 판별기에 입력
 gan = Model(ginput, dis_output)
 gan.summary()
 

 # GAN 훈련 함수
 def gan_train(epoch, batch_size, saving_interval):
  (X_train, _), (_, _) = mnist.load_data()
  # 데이터 범위 (0~255) 이미지 컬러 (흑백)
  X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype(
  'float32')
  X_train = (X_train - 127.5) / 127.5  # -1~1
  true = np.ones((batch_size, 1))  # 전부 1인 사이즈 32
  fake = np.zeros((batch_size, 1))  # 전부 0인 사이즈 32
 

  for i in range(epoch):
  idx = np.random.randint(0, X_train.shape[0],
  batch_size)
  imgs = X_train[idx]  # 실제 이미지 32장을 선택
  # 판별기를 학습
  d_loss_real = discriminator.train_on_batch(imgs, true)  # 배치 사이즈
  # 32장 x 100 노이즈 =>
  noise = np.random.normal(0, 1, (batch_size, 100))
  gen_imgs = generator.predict(noise)  # 32x28x28x1
  d_loss_fake = discriminator.train_on_batch(gen_imgs,
  fake)  # 판별기를 0으로
  # 두개의 loss가 반반씩 영향을 미침
  d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
  g_loss = gan.train_on_batch(noise, true)
  print('epoch:%d' % i, 'd_loss:%.4f' % d_loss,
  'g_loss:%.4f' % g_loss)
  if i % saving_interval == 0:
  noise = np.random.normal(0, 1, (25, 100))  # 25장 이미지를 만들기 위해서
  gen_imgs = generator.predict(noise)  # 가짜 이미지를 생성
  # 생성된 이미지를 원래의 사이즈
  gen_imgs = 0.5 * gen_imgs + 0.5  # 부동소수점 이미지인 0~1사이로 만
  fig, axs = plt.subplots(5, 5)
  count = 0
  for j in range(5):
  for k in range(5):
  axs[j, k].imshow(gen_imgs[count, :, :, 0],
  cmap='gray')
  axs[j, k].axis('off')
  count += 1
  fig.savefig("gan_images/gan_mnist_%d.png" % i)
 

 # GAN 훈련
 gan_train(4001, 32, 200)

 ###   결론

 오토인코더 및 GAN과 같은 비지도 학습 기법은 레이블이 지정되지 않은 데이터에서 의미 있는 정보를 추출하는 강력한 도구입니다. 오토인코더는 차원 축소, 노이즈 제거 및 이상 감지와 같은 작업에 탁월하며, GAN은 새롭고 사실적인 데이터를 생성할 수 있습니다. 이러한 기술은 이미지 처리, 데이터 분석 등 다양한 분야에서 광범위하게 응용됩니다.