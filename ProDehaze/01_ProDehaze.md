# 01_ProDehaze explanation

[논문](chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://arxiv.org/pdf/2503.17488)

### 사용한 traning data set
REalistic Single Image Dehazing (RESIDE):
    → 안개 낀 이미지와 깨끗한 원본 이미지 쌍
I-Haze, O-Haze, DenseHaze, NH-Haze 등
    → 전부 paired dataset
---
## 요약

1. 입력 이미지에서 내부 priors를 추출하여 사전 학습된 LDM(Latent Diffusion Model)이 중요한 영역에 집중하도록 프롬프트 제공
2. Structure-Prompted Restorer를 통해 구조가 뚜렷한 영역에 집중하여 디테일 충실도를 높임
    (Structure-Prompted Restorer : 구조가 잘 보이는 영역을 강조해서 latent space에서 보존하게 유도)
3. Haze-aware Refiner를 통해 깨끗한 영역과 출력 이미지 간의 정렬을 유도하고 색편향을 줄임
    (Haze-aware Refiner : 비교적 덜 흐린 영역을 기준으로 더 흐린 부분을 보정하도록 유도)
prompting으로 단순히 입력 이미지 외에 부가적인 정보를 줌
---
## 문제 제기

물리적 접근(Atmospheric Scattering Model(ASM) 기반)으로 안개의 두께, 조도 등을 물리적 parameter로 추정을 한다고 하면 정확한 파라미터 추정이 매우 어렵고, 실제 환경에서는 잘 안 맞음
딥러닝 모델로 hazzy -> clean 이미지를 mapping 학습하면 대부분 합성된 haze 데이터로 학습됨. 그래서 실제 안개 사진에는 일반화 성능이 약함

초기 연구들은 DM을 처음부터 학습했기 때문에, 사전학습된 생성 priors의 강력한 표현력을 활용하지 못했음. 이후에 사전학습된 LDMs을 복원에 활용해서 더 자연스럽고 사실적인 이미지 복원을 할 수 있게 됨. 그러나 실제 환경의 디헤이징에서는 hallucination 문제가 자주 발생하게 됨.
    (hallucination 문제 : 사전학습된 LDM은 외부 Data set에서 학습된 "자연 이미지 분포"를 기반으로 하므로, 실제 입력 이미지의 세부사항과 맞지 않는 구조나 색을 만들어내는 경우가 많음.)

prompt learning은 원래 NLP에서 개발된 것이지만, 최근에는 low-level vision tasks에도 적용되어 task 특화 priors를 활용할 수 있게 됨. 그러나 이러한 priors는 입력 이미지 내의 핵심 영역을 선택적으로 식별해 사전학습된 모델의 집중을 유도하는 능력이 부족하여, 그 효과가 희석되는 경우가 많음.
---
## 문제 해결방안

해당 논문은 이러한 문제를 해결하고자, 다음과 같은 2가지 종류의 prompt를 설계함.
1. Latent 공간에서, 구조가 정교한 영역이 우선시되도록 Structure-prompted Restorer를 사용함
2. 디코딩 중에는, 안개가 심한 영역은 후순위로 처리하고, Haze-aware Self-correcting Refiner를 통해 보다 신뢰성 있는 영역에 집중함.

![전체 구조](./images/fig1.png)
## Method

##Structure-Prompted Restorer(SPR, 구조 프롬프트 복원기)
사전학습된 LDM은 rich semantics(풍부한 의미)를 Lantent 공간에 인코딩을 함. 이 공간에서 선택적인 내부 priors, 즉 fine structures(미세 구조), contours(윤곽선), edges(엣지) 등의 시각적 단서가 복원에 중요함. 이러한 구조들은 이미지 도메인보다는 주파수 도메인에서 더 명확하게 분리됨. 그래서 high-frequency prompts(HFE, 고주파 프롬프트)를 dehazing 과정의 가이드로 사용함.

### HFE 추출
**x_high = conv₁×₁ [ (GLH * x_in) ⊕ (GHH * x_in) ⊕ (GHL * x_in) ]**
![alt text](./images/image-1.png)
(*는 컨볼루션, ⊕는 채널 단위 연결을 의미함)
1. Haar DWT(Discrete Wavelet Transform)
    입력 이미지 x_in에서  GLH, GHH, GHL 이 3개의 고주파 성분을 각각 Haar 필터를 사용해 방향별로 추출함. (Haar 필터 : 이미지의 경계나 변화(에지)를 빠르게 감지하는 필터)
2. Point-wise Convolution(1x1 ConV)
    3개의 고주파 성분을 채널 방향으로 concatenate 후 1x1 ConV를 적용해 x_high라는 고주파 특징 생성.

LDM의 denoising UNet이 x_high의 구조 정보를 활용하기 위해서 입력 이미지 x_in과 고주파 특징 x_high를 latent spcae에서 encoding한 뒤, 두 벡터를 결합하여 조건 벡터 c_f를 만듦.
**c_f = E(x_in) ⊕ E(x_high)**
![alt text](./images/image-2.png)

(E는 VAE 인코더)
c_f를 기존 denoising UNet의 사본을 trainable adaptor N의 입력으로 넣어 구조 프롬프트 injection함.

이 구조 프롬프트 조건 벡터 c_f를 기반으로 DM을 학습하며, 다음의 손실 함수를 최적화함. 학습 과정에서, conv 1x1 커널과 adaptor N만 학습하며, ε_θ는 고정됨.

**L_SPR = E_{x_in, t, c_f, ε ∼ N(0,1)} [‖ε − ε_θ(z_t, t, N(c_f))‖²₂]**
![alt text](./images/image-3.png)

(ε_θ는 사전학습된 denoising UNet, N(c_f)는 학습 가능한 adaptor가 주입한 조건)

## Haze-Aware Self-Correcting Refiner(HCR, 안개 인식 자기 교정 리파이너)

latent 공간에서 SPR로 복원할 결과는 충실도가 높지만, 최종 복원 이미지에서는 여전히 ground truth와의 불일지 발생 가능. -> hallucination 문제
이러한 문제를 해결하기 위해, 디코더를 fine-tuning하면서도 self-correction(자기 교정) 메커니즘을 설계함.

입력 이미지 내 더 깨끗한 영역과 출력 이미지 간의 정렬을 유도하고, 안개가 심한 영역의 영향은 줄이는 것이다! 이러한 과정을 위해서 우리는 Dark Channel Prior(DCP)를 이용해 안개 밀도를 추정한 뒤, 이 정보를 기반으로 attention map을 조정하는 방법을 사용함. 이후 디코더의 Window Swin Transformer(WST) 블록에서 self-attention을 haze-aware prior로 조절하면 더 정확하고 왜곡(색 왜곡)이 적은 복원 이미지 생성이 가능해짐. 
(DCP : 안개가 낀 이미지에서는 모든 픽셀이 희뿌옇다. 하지만 원래의 클린 이미지에서는 어떤 채널이든 한 군데 쯤은 매우 어두운 값(0에 가까운 값)이 있어야 자연스럽다.라는 원리를 사용한 이미지 dehazing 방식

Window Swin Transformer(WST) : 윈도우(작은 블록)를 기준으로 attention을 하되, 윈도우 위치를 번갈아가며 이동시키는 방식)

**M_DCP ∈ ℝ^{N×1}, N = H × W**

![alt text](./images/image-5.png)

(H,W는 입력 이미지 x_in의 높이와 너비)
위의 식은 haze density(안개 농도)를 일렬로 펴서 1차원 벡터로 변환한거임.

**M_corr,l = (M_DCP * W_Q^l) * (M_DCP * W_K^l)^T**

![alt text](./images/image-4.png)

(W_Q^l ∈ ℝ^{1×N_l}, W_K^l ∈ ℝ^{1×N_l}: 학습 가능한 가중치 행렬, N_l = H_l × W_l: 디코더의 l번째 블록에 해당하는 feature map 크기)

**I={(i,j)∣M_ij^corr,l ∈ Topk(M_corr,l)}**

![alt text](./images/image-7.png)

M_corr,l에서 가장 높은 값 상위 k개를 선택하여 인덱스 집합 I를 생성함.

![alt text](./images/image-6.png)

선택된 상위 k개 위치만 남기고 나머지는 억제하는 sparse mask(희소 마스크) M_s를 구성함.
즉, 안개가 많은 영역은 영향력을 제거하고, 안개가 적은 영역은 1 - corr값으로 강조함.

**attn(M_s) = softmax(((QK^T)⊙M_s / N_l))V**

![alt text](./images/image-8.png)

(Q,K,V : attention 연산의 Query, Key, Value, ⊙ : 요소별 곱)
WST 블록의 attention 계산식에 위의 M_s를 적용하여 attention을 안개 농도에 따라 조절함.

**x_r = D(z_0;R(F_E(x_in),F_D),attn(M_s))**

![alt text](./images/image-9.png)

HCR은 디코더의 attention을 haze-aware prior로 조정하고, R이라는 학습 가능한 refine network를 통해 encoder의 feature F_E(x_in)와 decoder의 feature F_D를 정렬하여 다음과 같은 최종 출력 이미지 x_r를 생성함.

**L_HCR = ∥x_r − x_GT∥_1 + L_VGG(x_r,x_GT) + L_adv(x_r,x_GT)**

![alt text](./images/image-10.png)

(x_GT : ground truth 이미지, L_VGG : VGG perceptual loss, L_adv : adversarial loss)
(L_VGG : 사람 눈에 보이는 고수준 시각적 유사도를 반영하기 위해, VGG 네트워크의 중간 feature를 기준으로 두 이미지의 차이를 계산하는 loss 함수(CNN 기반)
L_adv : 판별자를 속이기 위해 생성자가 출력한 이미지가 진짜같이 보이도록 학습시키는 loss 함수(GAN 기반))
최종 HCR 학습 시, 다음과 같은 손실 함수를 최소화함.
---
## Experiments



---
## 궁금증

1. 안개 정도가 다 다른데 처음 본 안개 낀 이미지를 어떻게 복원 시킬 수 있는 것인가?
    SPR에서 x_in의 고주파 성분(윤곽)을 Haar Filter로 뽑아낸 x_high 덕분이다.
    HCR에서 x_in 자체에서 안개가 덜 낀 부분을 DCP로 판단해서 attention을 덜 뿌연 곳에 더 주도록 재가중함. 즉, 입력에서 신뢰 가능한 부분만 우선 참고하는 것임.
    **구조 + attention + 조건화를 통해 스스로 전략을 세움. 또한, ProDehaze는 학습 시 여러 안개 레벨을 포함한 데이터로 훈련되기 때문에 그 패턴에 맞춰 다양한 정도의 x_in이 들어와도 문제없이 복원 가능함**


