---
title: Matplotlib 시각화
date: 2025-02-19 1:11:00 +09:00
categories: [Python]
tags: [Python, NumPY, Matplotlib]
---

# 데이터 시각화: 정보를 효과적으로 표현하는 방법  

데이터 시각화는 **숫자, 텍스트(ASCII 코드), 이미지(RGBA), 사운드, 동영상** 등의 정보를 그래픽 요소로 표현하는 기법입니다.  
이를 통해 **시간, 추세, 분포, 관계, 비교, 공간(GIS)** 등의 패턴을 효과적으로 분석할 수 있습니다.  

---

## 🔹 데이터 시각화의 핵심 개념  

### ✅ 회귀 분석과 시각화  
회귀 분석을 수행할 때 데이터의 특성을 파악하기 위해 다음과 같은 시각화 기법을 활용할 수 있습니다.  

- **정규성 확인**: Q-Q plot (Quantile-Quantile Plot)  
- **등분산성 확인**: Residual Plot (잔차 그래프)  
- **독립성 확인**: Time Series Plot (시계열 그래프)  
- **선형성 확인**: Scatter Plot (산점도)  

```python
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Q-Q plot 예제 (정규성 확인)
data = np.random.normal(0, 1, 1000)  # 정규분포 데이터 생성
stats.probplot(data, dist="norm", plot=plt)
plt.show()
```

---

## 🔹 데이터 정보 표현 방식  

데이터를 시각화할 때 다양한 방식으로 정보를 전달할 수 있습니다.  

| 표현 방식 | 설명 |
|-----------|--------------------------------------------|
| **Correlation** | 변수 간의 관계를 표현 |
| **Deviation** | 평균값에서의 편차를 강조 |
| **Ranking** | 데이터의 순위를 나타냄 |
| **Distribution** | 데이터의 분포를 시각적으로 표현 |
| **Composition** | 데이터의 구성 요소 비율 표시 (예: 원형 그래프) |
| **Change** | 시간에 따른 변화를 나타냄 |
| **Groups** | 데이터의 그룹별 차이를 강조 |

---

## 🔹 Matplotlib 그래프의 구성 요소  

Matplotlib을 이용한 시각화는 다음과 같은 요소로 구성됩니다.  

- **Figure** : 전체 그래프를 그리는 **캔버스**  
- **Axes** : 그래프의 **도화지**  
- **Axis** : x축, y축을 의미  
- **Spines** : 그래프의 테두리  
- **그래픽 객체** : 시각화 요소 (점, 선, 면, 폰트, 컬러 등)  
- **속성 설정** : 마커, 레이블, 범례, 축 속성 조정  

```python
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [4, 5, 6])  # 기본 라인 그래프
plt.show()
```

---

## 🔹 Matplotlib 시각화 종류  

Matplotlib에서는 **이산적 데이터**와 **연속적 데이터**를 시각화하는 다양한 방법을 제공합니다.  

### ✅ **이산적 데이터(Discrete Data) 시각화**  

| 함수 | 설명 |
|------|-------------------------------------|
| `bar()` | 막대 그래프 |
| `barh()` | 가로 막대 그래프 |
| `pie()` | 원형 그래프 |
| `scatter()` | 산점도 (점 그래프) |

```python
# 막대 그래프 예제
x = ['A', 'B', 'C', 'D']
y = [10, 20, 15, 25]

plt.bar(x, y)
plt.show()
```

### ✅ **연속적 데이터(Continuous Data) 시각화**  

| 함수 | 설명 |
|------|-------------------------------------|
| `plot()` | 선 그래프 |
| `hist()` | 히스토그램 (빈도 그래프) |
| `boxplot()` | 박스 플롯 (분포 확인) |
| `violinplot()` | 바이올린 플롯 (분포 확인) |
| `scatter()` | 산점도 |
| `pcolormesh()`, `imshow()` | 행렬 기반의 색상 그래프 |

```python
# 히스토그램 예제
data = np.random.randn(1000)

plt.hist(data, bins=30)
plt.show()
```

### ✅ **3D 시각화**  

| 함수 | 설명 |
|------|-------------------------------------|
| `scatterplot` | 3D 산점도 |
| `3d surface` | 3D 표면 그래프 |
| `wireframe` | 3D 와이어프레임 |

```python
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = np.random.rand(100)
y = np.random.rand(100)
z = np.random.rand(100)

ax.scatter(x, y, z)
plt.show()
```

---

## 🔹 2D vs 3D 시각화  

- **2D 시각화** : 두 개의 값(x, y)이 필요하지만, 하나만 입력되면 y값으로 간주됨  
- **3D 시각화** : x, y, z 세 개의 값이 필요하며, 공간적인 관계를 나타낼 때 사용  

---

## 🎯 데이터 시각화 활용  

✔ 데이터의 분포 및 관계 파악  
✔ 시계열 데이터 분석  
✔ 비교 및 변화 추세 파악  
✔ 공간 데이터(GIS) 분석  

데이터 시각화는 **데이터를 더욱 직관적으로 이해하고, 효과적인 인사이트를 도출하는 필수 기술**입니다.  
Matplotlib을 활용하여 다양한 데이터 시각화를 시도해 보세요!  

---
