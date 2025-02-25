---
title: 데이터 전처리리
date: 2025-02-21 12:11:00 +09:00
categories: [Python]
tags: [Python, 데이터 전처리리]
---

## 데이터 전처리 기법

데이터 전처리는 데이터 분석 및 모델링에 앞서 데이터를 정제하고 변환하는 필수적인 과정입니다. 이를 통해 데이터 품질을 향상시키고 분석 결과의 정확성을 높일 수 있습니다. 주요 데이터 전처리 기법은 다음과 같습니다.

## 1. 결측치 처리

결측치는 데이터에 값이 없는 경우를 말합니다. 결측치를 처리하는 방법은 다양하며, 데이터의 특성과 분석 목적에 따라 적절한 방법을 선택해야 합니다.

* **제거:** 결측치가 있는 행 또는 열 전체를 제거합니다.
* **대체:** 결측치를 다른 값으로 대체합니다. 평균, 중앙값, 최빈값 등을 사용할 수 있습니다.
* **보간:** 결측치를 주변 값을 사용하여 추정합니다. 선형 보간, 다항식 보간 등 다양한 방법이 있습니다.

```python
import pandas as pd
import numpy as np

# 데이터 생성
df = pd.DataFrame({'A': [1, 2, np.nan, 4, 5],
                   'B': [6, np.nan, 8, 9, 10]})

# 결측치 제거
df_dropna = df.dropna()
print(df_dropna)

# 결측치 대체 (평균으로 대체)
df_fillna = df.fillna(df.mean())
print(df_fillna)

# 결측치 보간 (선형 보간)
df_interpolate = df.interpolate()
print(df_interpolate)
```

## 2. 이상치 처리

이상치는 다른 데이터와 크게 차이가 나는 값을 말합니다. 이상치는 분석 결과에 큰 영향을 미칠 수 있으므로, 적절하게 처리해야 합니다.

* **제거:** 이상치를 제거합니다.
* **변환:** 이상치를 다른 값으로 변환합니다. 로그 변환, 제곱근 변환 등을 사용할 수 있습니다.
* **Winsorizing:** 이상치를 특정 백분위수 값으로 대체합니다.

```python
from scipy import stats

# z-score를 이용한 이상치 제거
data = pd.Series(np.random.normal(size=100))
z_scores = stats.zscore(data)
abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 3).all(axis=1)  # z-score 3을 초과하는 값 제거
data_filtered = data[filtered_entries]
print(data_filtered)
```

## 3. 데이터 변환

데이터 변환은 데이터 분석에 적합하도록 데이터 형식을 변경하는 과정입니다.

* **스케일링:** 데이터 범위를 조정합니다. 표준화, 정규화 등 다양한 방법이 있습니다.
* **인코딩:** 범주형 데이터를 숫자형 데이터로 변환합니다. 원-핫 인코딩, 레이블 인코딩 등이 있습니다.
* **집계:** 데이터를 요약합니다. 합계, 평균, 개수 등을 계산할 수 있습니다.

```python
from sklearn.preprocessing import StandardScaler

# 표준화
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data.reshape(-1, 1))
print(data_scaled)
```

## 4. 데이터 축소

데이터 축소는 데이터 크기를 줄이는 과정입니다.

* **차원 축소:** 변수 개수를 줄입니다. 주성분 분석, 선형 판별 분석 등이 있습니다.
* **샘플링:** 데이터 일부를 추출합니다. 랜덤 샘플링, 계층적 샘플링 등이 있습니다.

## 5. 데이터 정제

데이터 정제는 데이터 오류를 수정하는 과정입니다.

* **오타 수정:** 오타를 수정합니다.
* **중복 제거:** 중복된 데이터를 제거합니다.
* **형식 통일:** 데이터 형식을 통일합니다.

```python
# 중복 제거
data_unique = data.unique()
print(data_unique)
```

## 6. 추가적인 전처리 기법

* **텍스트 데이터 전처리:** 텍스트 데이터를 분석에 적합하게 변환합니다. 토큰화, 불용어 제거, 어간 추출 등이 있습니다.
* **시계열 데이터 전처리:** 시계열 데이터를 분석에 적합하게 변환합니다. 차분, 이동 평균, 지수 평활법 등이 있습니다.

---

