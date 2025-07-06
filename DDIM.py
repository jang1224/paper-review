import math
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from keras import layers

dataset_name="oxford_flowers102" # 옥스퍼드 꽃 데이터셋
dataset_repetitions=5
img_siz=64
batch_siz=64

kid_img_siz=75 # KID
kid_diffusion_steps=5
plot_diffusion_steps=20

min_signal_rate=0.02 # 샘플링
max_signal_rate=0.95

zdim=32 # 싞경망 구조
embed_max_freq=1000.0
widths=[32,64,96,128]
block_depth=2

def preprocess_image(data):
    height=tf.shape(data["image"])[0] # 중앙 잘라내기(center cropping)
    width=tf.shape(data["image"])[1]
    crop_siz=tf.minimum(height, width)
    image=tf.image.crop_to_bounding_box(data["image"],(height-crop_siz)//2,(width-crop_siz)//2,crop_siz,crop_siz)
    image=tf.image.resize(image,size=[img_siz,img_siz],antialias=True) # antialias=True 설정
    return tf.clip_by_value(image/255.0,0.0,1.0)

def prepare_dataset(split):
    return (tfds.load(dataset_name,split=split,shuffle_files=True)
    .map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE).cache()
    .repeat(dataset_repetitions).shuffle(10*batch_siz) # 셔플링은 KID에 중요
    .batch(batch_siz,drop_remainder=True)   # batch_siz는 한번에 몇 장의 이미지로 돌릴거냐의 의미. drop_remainder=True이면 batch_siz로 나눠지지 않는 건 버림.
    .prefetch(buffer_size=tf.data.AUTOTUNE))   

train_dataset=prepare_dataset("train[:80%]+validation[:80%]+test[:80%]") # 데이터셋
val_dataset=prepare_dataset("train[80%:]+validation[80%:]+test[80%:]")

class KID(keras.metrics.Metric): # KID(Kernel Inception Distance) 측정을 위한클래스
    def __init__(self,name,**kwargs):
        super().__init__(name=name,**kwargs)
        self.kid_tracker=keras.metrics.Mean(name="kid_tracker") 
        self.encoder=keras.Sequential( # InceptionV3 사용
        [keras.Input(shape=(img_siz,img_siz,3)),layers.Rescaling(255.0),
        layers.Resizing(height=kid_img_siz,width=kid_img_siz),
        layers.Lambda(keras.applications.inception_v3.preprocess_input),
        keras.applications.InceptionV3(include_top=False,input_shape=(kid_img_siz,kid_img_siz,3),weights="imagenet"),
        layers.GlobalAveragePooling2D()],name="inception_encoder")

    def polynomial_kernel(self,features_1,features_2):
        feature_dimensions=tf.cast(tf.shape(features_1)[1],dtype=tf.float32)
        return (features_1 @ tf.transpose(features_2)/feature_dimensions+1.0)**3.0

    def update_state(self,real_images,generated_images,sample_weight=None):
        real_features=self.encoder(real_images,training=False)
        generated_features=self.encoder(generated_images,training=False)

        kernel_real=self.polynomial_kernel(real_features,real_features) # 두 특징으로 다항식
        kernel_generated=self.polynomial_kernel(generated_features,generated_features)
        kernel_cross = self.polynomial_kernel(real_features, generated_features)

        batch_siz=tf.shape(real_features)[0] # 평균 커널값으로 squared maximum mean 
        batch_sizf=tf.cast(batch_siz,dtype=tf.float32) #discrepancy 측정
        mean_kernel_real=tf.reduce_sum(kernel_real*(1.0-tf.eye(batch_siz)))/(batch_sizf*(batch_sizf-1.0))
        mean_kernel_generated=tf.reduce_sum(kernel_generated*(1.0-tf.eye(batch_siz)))/(batch_sizf*(batch_sizf-1.0))
        mean_kernel_cross=tf.reduce_mean(kernel_cross)
        kid=mean_kernel_real+mean_kernel_generated-2.0*mean_kernel_cross
        self.kid_tracker.update_state(kid) # 평균 KID 측정을 갱신

    def result(self):
        return self.kid_tracker.result()

    def reset_state(self):
        self.kid_tracker.reset_state()

def sinusoidal_embedding(x):    #timestep을 인코딩하기 위해서 사용
    embed_min_freq=1.0
    freq=tf.exp(tf.linspace(tf.math.log(embed_min_freq),tf.math.log(embed_max_freq),zdim//2))
    angular_speeds=2.0*math.pi*freq
    embeddings=tf.concat([tf.sin(angular_speeds*x),tf.cos(angular_speeds*x)],axis=3)        #sin이 zdim//2개, cos이 zdim//2개 총 zdim차원의 채널이 출력을 나옴
    return embeddings
# REsidualBlock은 입력과 출력의 차이를 학습하는 구조
def ResidualBlock(width):       # 3x3 convolution-> swish-> skip 연결(ResNet 스타일)
    def apply(x):
        input_width=x.shape[3]
        if input_width==width: residual=x
        else: residual=layers.Conv2D(width,kernel_size=1)(x)  # x랑 conV 결과랑 채널의 크기가 같아야 하므로 1x1 conV를 통해 잔차연결할 residual를 conV의 출력과 같은 채널 크기로 맞춰주기  

        x=layers.BatchNormalization(center=False,scale=False)(x)    #BatchNormalization : 평균과 분산을 조정하는 과정이 별도의 과정으로 떼어진 것이 아니라, 신경망 안에 포함되어 학습 시 평균과 분산을 조정하는 과정
        x=layers.Conv2D(width,kernel_size=3,padding="same",activation=keras.activations.swish)(x)   #swish(x) = x * sigmoid(x), 활성화함수로 swish를 쓰는 이유 : 부드럽고 학습 안정적이여서
        x=layers.Conv2D(width,kernel_size=3,padding="same")(x)
        x=layers.Add()([x,residual])        # 입력과 변화된 정보 더해주기
        return x

    return apply

def DownBlock(width,block_depth):       # 다운샘플링 (AveragePooling2D)
    def apply(x):
        x,skips=x
        for _ in range(block_depth):
            x=ResidualBlock(width)(x)
            skips.append(x)     # 나중에 업샘플링할 때 이 skips들을 꺼내서 concat or add함
        x=layers.AveragePooling2D(pool_size=2)(x)   # 이미지의 해상도를 반으로 줄이기 위해 2x2 평균 풀링 사용
        return x

    return apply

def UpBlock(width,block_depth):     # 업샘플링 ( Concatenate + ResidualBlock)
    def apply(x):
        x,skips=x
        x=layers.UpSampling2D(size=2,interpolation="bilinear")(x)       # H와 W를 2배로 키운다. bilinear는 픽셀 사이 값을 부드럽게 추정해서 이미지를 자연스럽게 키우는 방식(부드럽게 보간)
        for _ in range(block_depth):
            x=layers.Concatenate()([x,skips.pop()])     # 업샘플링된 feature map이랑 skip feature map을 채널 차원에서 결합
            x=ResidualBlock(width)(x)
        return x

    return apply

def get_network(image_size,widths,block_depth):     # U-Net의 전체 구성 + 시간 임베딩과 noisy image 통합
    noisy_images=keras.Input(shape=(image_size,image_size,3))   # 디퓨전 단계에서 노이즈가 더해진 이미지
    noise_variances=keras.Input(shape=(1,1,1))  #해당 이미지가 어느 timestep인지 나타내는 정보( 보통 분산임 )

    e=layers.Lambda(sinusoidal_embedding)(noise_variances)  #1x1xzdim 크기가 됨 / noise_variances라는 입력 텐서를 sinusoidal_embedding()함수를 통과시켜서 시간 정보를 인코딩한 임베딩 벡터로 바꾸는 작업
    e=layers.UpSampling2D(size=image_size,interpolation="nearest")(e)   # image_size x image_size x zdim 크기가 됨 / noisy_images랑 concat해주기 위해서 image_size로 업샘플링 해준거임

    x=layers.Conv2D(widths[0],kernel_size=1)(noisy_images)  # 채널의 크기는 widths[0] = 32차원임
    x=layers.Concatenate()([x,e])   # 채널 수 맞췄으니 x와 e를 concat해줌

    skips=[]
    for width in widths[:-1]:
        x=DownBlock(width,block_depth)([x,skips])   # 마지막 인덱스의 차원빼고 인덱스 0번째의 차원부터 맞추고 다운샘플링을 계속 해줌
    for _ in range(block_depth):
        x=ResidualBlock(widths[-1])(x)  #제일 큰 차원을 width로 넣어주고 고정된 해상도 & 채널 크기에서 깊이 있는 연산을 반복해서 특징 더 뽑아내기
    for width in reversed(widths[:-1]): # 마지막 인덱스의 차원 빼고 widths에 있는 큰 차원부터 역으로 채널의 크기만큼 업샘플링 시킨다. -> 원래 차원의 채널로 바뀐다
        x=UpBlock(width,block_depth)([x,skips])
    x=layers.Conv2D(3,kernel_size=1,kernel_initializer="zeros")(x)      #초기 출력을 0으로 하는 이유 : 삭습을 통해 점점 의미 있는 이미지를 만들도록 하기 위해 (아무것도 하지마라는 의미)

    return keras.Model([noisy_images,noise_variances],x,name="residual_unet")   # [노이즈가 섞인 이미지, 각 이미지에 해당하는 노이즈 스케일 값], 최종적으로 예측된 노이즈 or 복원 이미지(여기서는 RGB 이미지 형태로 출력됨), model 이름(residual_unet) 

class DiffusionModel(keras.Model): # 확산 모델을 위한 클래스
    def __init__(self,image_size,widths,block_depth):
        super().__init__()
        self.normalizer=layers.Normalization()      # 층 정규화 -> Standard 정규화(Z-score)로 평균 0, 표준편차 1로 바꾸는 방식(가우시안 분포를 따름)
        self.network=get_network(image_size,widths,block_depth) # denoise용 U-net   -> 가중치가 계속 gradient descent로 업데이트됨
        self.ema_network=keras.models.clone_model(self.network) # KID용 U-net   -> 가중치가 EMA(지수이동평균)로 업데이트 됨
        # KID같이 평가할 때 학습 중인 네트워크는 불안정할 수 있어서 과거 가중치의 평균적인 추세를 따르는 ema 네트워크를 사용함
    def compile(self, **kwargs):
        super().compile(**kwargs)
        self.noise_loss_tracker = keras.metrics.Mean(name="n_loss") # keras.metrics.Mean은 여러 값을 받아서 평균을 자동으로 계산해주는 유틸 클래스.
        self.image_loss_tracker = keras.metrics.Mean(name="i_loss")
        self.kid = KID(name="kid")

    @property   #method를 클래스의 속성처럼 사용할 수 있게 해줌. 함수 호출 없이 객체.속성처럼 접근할 수 있게 해줌
    def metrics(self):
        return [self.noise_loss_tracker, self.image_loss_tracker, self.kid]

    def denormalize(self, images): # 화소 값을 [0,1] 사이로 역변환(normalized = (image - mean) / sqrt(variance) -> 이 공식을 이용한거임)
        images = self.normalizer.mean + images * self.normalizer.variance**0.5
        return tf.clip_by_value(images,0.0,1.0)     # 역변환된 이미지가 float 연산 오류 등으로 인해 [0,1] 범위를 벗어날 수 있기 때문에 [0.0,1.0] 범위로 잘라준거임
    # diffusion_schedule은 DDPM이 아닌 DDIM에서만 쓰이는 방식
    def diffusion_schedule(self, diffusion_times):      # diffusion_tiems으로 들어온 [0,1] 범위 값에 대해 각도로 변환 후, singal_rate와 noise_rate 계산 실행한거임
        start_angle=tf.acos(max_signal_rate) # 확산 시간을 각도로 변환
        end_angle=tf.acos(min_signal_rate)
        diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle) # 각 timestep에 대해 signal_rate와 noise_rate를 조절해야 해서 선형 매핑한거임
        # diffusion_times이 커질수록 signal_rates의 값은 0에 가까워지고, noise_rates는 1에 가까워진다.
        signal_rates=tf.cos(diffusion_angles) # signal_rates와
        noise_rates=tf.sin(diffusion_angles) # noise_rates의 제곱 합은 1
        return noise_rates,signal_rates
    def denoise(self, noisy_images, noise_rates, signal_rates, training):       # network 선택후 디노이징 실행
            
        if training: network=self.network # 학습할 때 쓰는 신경망
        else: network=self.ema_network # KID 평가할 때 쓰는 신경망

        pred_noises=network([noisy_images,noise_rates**2],training=training)    # 예측한 노이즈값
        pred_images=(noisy_images - noise_rates * pred_noises) / signal_rates # x₀ ≈ x₀_pred = (x_t - √(1 - α) * ε_pred) / √α
        return pred_noises,pred_images

    def reverse_diffusion(self,initial_noise,diffusion_steps):      # 순수한 노이즈에서 시작해 점점 이미지를 생성하는 과정
        num_images = initial_noise.shape[0]       # initial_noise는 가우시안 노이즈로 초기화된 이미지 tensor(xt부터 시작)
        step_size = 1.0 / diffusion_steps       # 몇 step으로 역확산할지 / [0,1]로 맞추고 t=1에서 t=0로 디퓨전 시간을 균등하게 나누기 위한 설정

        next_noisy_images = initial_noise # 매 스텝마다 업데이트될 이미지 ( 시작은 완전 노이즈 )
        for step in range(diffusion_steps):
            noisy_images = next_noisy_images
            # tf.ones((num_images,1,1,1))은 모든 이미지에 동일한 t값 주입용(broadcast)
            diffusion_times = tf.ones((num_images,1,1,1)) - step  * step_size       # 모든 이미지에 대해 동일한 diffusion time을 가질 수 있도록 정해진 스칼라 값을 각 이미지에 broadcast해주는 것
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)    # 현재 시점 t에 해당하는 노이즈 비율(sin)과 신호 비율(cos)를 계산한다.
            pred_noises,pred_images=self.denoise(noisy_images,noise_rates,signal_rates,training=False)
 
            next_diffusion_times = diffusion_times - step_size  # 다음 시점
            next_noise_rates, next_signal_rates = self.diffusion_schedule(next_diffusion_times) # 다음 시점의 noise랑 signla 비율을 계산
 
            next_noisy_images=(next_signal_rates * pred_images + next_noise_rates * pred_noises) # x_t-1 = √α * x₀_pred + √(1 - α) * ε_pred
        return pred_images  # x₀_pred return

    def generate(self,num_images,diffusion_steps):      # 이미지 생성을 위한 함수
        initial_noise=tf.random.normal(shape=(num_images, img_siz, img_siz, 3))     # 무작위 노이즈 이미지 생성
        generated_images=self.reverse_diffusion(initial_noise,diffusion_steps) # 노이즈로부터 이미지 복원 시작, diffusion_steps만큼 반복하여 t=1 -> t=0으로 진행
        generated_images=self.denormalize(generated_images) # 역정규화 / 복원된 이미지를 평균과 분산을 이용해 [0,1] 범위로 역변환
        return generated_images

    def train_step(self, images):   # 학습 step) forward pass -> loss 계산 -> backpropagation -> weight update 과정이 이루어짐
        images = self.normalizer(images,training=True) # 정규화
        noises = tf.random.normal(shape = (batch_siz,img_siz,img_siz,3)) # 잡음 (원본 이미지에 더할 랜덤 가우시안 노이즈(ε))
 
        diffusion_times = tf.random.uniform(shape = (batch_siz,1,1,1), minval=0.0, maxval=1.0)  # 무작위 diffusion 시간(t) 선택 
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)    # noise / signal rate 계산
        noisy_images = signal_rates * images + noise_rates * noises # 확산 스케쥴에 따라 잡음과 이미지 혼합 # x_t = √α * x₀_pred + √(1 - α) * ε_pred
        
        with tf.GradientTape() as tape: # 이 안에서 이루어지는 모든 연산 기록함 / denoise로 잡음과 영상 분리하고 손실 계산 -> MAE(L1) 사용함
            pred_noises,pred_images = self.denoise(noisy_images,noise_rates,signal_rates,training=True) # noisy image를 넣어 예측된 노이즈, 이미지 추출 / training=True니까 학습용 U-Net 사용함
            noise_loss = self.loss(noises,pred_noises) # 학습에 사용하는 손실->노이즈를 비교하는 거니까 학습
            image_loss = self.loss(images,pred_images) # 평가에 사용하는 손실->이미지를 비교하는 거니까 평가
        # noise_loss를 기준으로 self.network.trainable_weights에 대해 미분 수행(ex) ∂(loss) / ∂(weight_1), ∂(loss) / ∂(weight_2), ...)
        gradients=tape.gradient(noise_loss,self.network.trainable_weights)      # noise_loss를 기준으로 모든 weight에 대한 gradient 계산
        self.optimizer.apply_gradients(zip(gradients,self.network.trainable_weights))   # optimizer( AdamW )가 weight 업데이트 한다 / zip( list1,list2 )는 두 개의 리스트를 짝지어서 순회할 수 있게 해주는 파이썬 함수( ex) zip(['a', 'b'], [1, 2]) → ('a', 1), ('b', 2) )
        self.noise_loss_tracker.update_state(noise_loss)    # epoch마다 metric 업데이트
        self.image_loss_tracker.update_state(image_loss)    # epoch마다 metric 업데이트
        #평가 시 더 안정적인 결과를 위한 전략
        for weight,ema_weight in zip(self.network.weights, self.ema_network.weights):   # 학습 중인 가중치를 추적해서 EMA 비전 만들기 (지수 이동 평균)
            ema_weight.assign( 0.999 * ema_weight + (1 - 0.999) * weight ) # 가중치의 EMA 추적, EMA 방식으로 값 업데이트

        return {m.name:m.result() for m in self.metrics[:-1]}       # KID를 생략하고 현재 누적된 평균값(손실 or metric의 평균값(n_loss / i_loss))

    def test_step(self, images):    # validation(검증) or test 데이터셋에 대해 모델의 성능을 체크
        images=self.normalizer(images,training=False)       # images는 평균 0, 분산 1로 바뀜 -> why? noise가 가우시안 분포여서 이미지도 이와 비슷한 스케일을 가져야 학습이 안정됨
        noises=tf.random.normal(shape=(batch_siz,img_siz,img_siz,3))    # xT를 만드는 과정
        diffusion_times=tf.random.uniform(shape=(batch_siz,1,1,1),minval=0.0,maxval=1.0)    # 무작위 diffusion시간(t) 선택
        noise_rates,signal_rates=self.diffusion_schedule(diffusion_times)   # 무작위 diffusion 시간(t)에서의 noise와 signal의 rate return
        noisy_images = signal_rates * images + noise_rates * noises # 시간 t에서의 noise와 signal의 rate를 이용해서 x0와 xT의 비율을 정해주고 이를 noisy_images(xt임)에 저장함.
        # 예측 복원 이미지, 예측 노이즈 생성
        pred_noises,pred_images=self.denoise(noisy_images,noise_rates,signal_rates,training=False)  # training=False이니까 KID 평가용인 ema_network를 사용함
        noise_loss = self.loss(noises,pred_noises)      # 예측한 noise랑 정답 noise의 차이 loss를 noise_loss에 저장
        image_loss = self.loss(images,pred_images)      # 예측한 images랑 정답 images의 차이 loss를 image_loss에 저장
        self.image_loss_tracker.update_state(image_loss)    # 위에서 계산한 loss로 image_loos_tracker를 갱신 -> metric 갱신
        self.noise_loss_tracker.update_state(noise_loss)    # 위에서 계산한 loss로 noise_loos_tracker를 갱신 -> metric 갱신
        images = self.denormalize(images)   # 이미지가 가우시안 분포로 정규화 되어 있던것을 다시 픽셀값 스케일(밝기 값)로 복원 -> [0,1] 범위의 복원
        generated_images=self.generate(num_images=batch_siz, diffusion_steps = kid_diffusion_steps) # 새 이미지 생성
        self.kid.update_state(images, generated_images) # 정답 image랑 새로 생성한 image 비교해서 KID 계산
        return {m.name: m.result() for m in self.metrics}   #KID를 포함한 현재 누적된 평균값(손실 or metric의 평균값(n_loss / i_loss))
    #그리기
    def plot_images(self,epoch=None,logs=None,num_rows=1,num_cols=8): # 영상 생성하고
        generated_images=self.generate(num_images = num_rows * num_cols, diffusion_steps = plot_diffusion_steps)   # num_rows * num_cols 개수의 이미지 생성

        plt.figure(figsize=(num_cols * 2.0, num_rows * 2.0))    #그림판 열기, 전체 subplot의 크기 지정
        for row in range(num_rows):
            for col in range(num_cols):
                index = row * num_cols + col
                plt.subplot(num_rows,num_cols,index+1)  # 위치 설정
                plt.imshow(generated_images[index])     # 이미지 보여주기
                plt.axis("off")     # 축 제거
                plt.tight_layout()  # subplot끼리 겹치지 않게 정렬
                plt.show()      # 화면에 띄우기
                plt.close()     # 리소스 정리(메모리 누수 방지)

model=DiffusionModel(img_siz, widths, block_depth) # 1. 모델 생성

cp_path="checkpoints/diffusion_model" # 2. 체크포인트: 최고 모델 저장(KID 메트릭 사용)
cp_callback=tf.keras.callbacks.ModelCheckpoint(filepath=cp_path, save_weights_only=True, monitor="val_kid", mode="min", save_best_only=True)

model.normalizer.adapt(train_dataset) # 3. 모델이 train_dataset의 평균과 분산을 자동으로 학습하고, 그 후 학습 중에 이 평균과 분산을 사용해 정규화함
# AdamW에서 learning_rate = 1000, weight_decay = 0.0001임
model.compile(optimizer=tf.keras.optimizers.experimental.AdamW(learning_rate=1e3,weight_decay=1e-4), loss=keras.losses.mean_absolute_error)      # 4. optimizer로 AdamW를 쓰고, loss함수로는 MAE(L1)를 씀
model.fit(train_dataset, epochs=50, validation_data=val_dataset, callbacks=[keras.callbacks.LambdaCallback(on_epoch_end=model.plot_images),cp_callback])   # 5. 학습 시작 : epoch마다 이미지 생성을 하고, KID 기준 체크포인트 저장도 함

model.load_weights(cp_path) # 6. 최고 성능 모델 불러오기 (KID가 가장 낮았던 시점)
model.plot_images() # 7. 이미지 생성해서 시각적으로 확인