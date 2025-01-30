---
title: tidyverse, SQL
date: 2025-01-30 05:28:00 +09:00
categories: [R, 데이터]
tags: [R, R Studion, 통계, tidyverse, SQL, 데이터 전처리]
---
### 들어가기에 앞서
- [R을 통한 데이터 정규화와 전처리 이해하기](https://parkyaeseong.github.io/posts/R%EC%9D%84-%ED%86%B5%ED%95%9C-%EB%8D%B0%EC%9D%B4%ED%84%B0-%EC%A0%95%EA%B7%9C%ED%99%94%EC%99%80-%EC%A0%84%EC%B2%98%EB%A6%AC-%EC%9D%B4%ED%95%B4%ED%95%98%EA%B8%B0/)를 보고 오시는 것을 추천 드립니다.


## Tidyverse 주요 패키지 소개
- **tidyverse**: 데이터 과학을 위한 통합 패키지 집합
- **magrittr**: `%>%` 파이프 연산자 제공 (Ctrl + Shift + M)
- **dplyr**: 데이터 변환 핵심 함수들
- **tidyr**: 데이터 구조 변환 (long/wide 형식)
- **readr**: 데이터 읽기/쓰기
- **stringr**: 문자열 처리
- **forcats**: 범주형 데이터 처리
- **purrr**: 함수형 프로그래밍
- **lubridate**: 날짜/시간 처리
- **ggplot2**: 데이터 시각화
- **dbplyr**: 데이터베이스 연동

## dplyr 주요 함수
```r
library(tidyverse)

# 데이터 생성
data <- data.frame(
  x1 = 1:6,
  x2 = c(1,2,2,3,1,2),
  x3 = c("F", "B", "C", "E", "A", "D")
)

# 1. 열 선택
dplyr::select(data, x1, x2)
dplyr::select(data, -x1)

# 2. 정렬
arrange(data, x3)
arrange(data, desc(x1))

# 3. 필터링
filter(data, x2 == 2)
filter(data, x1 > 4 & x2 > 1)

# 4. 파생변수 생성
data <- mutate(data, 
              x4 = x1 + x2,
              x5 = x4/100,
              x6 = x4*x2)

# 5. 요약 통계
summarise(data, avg = mean(x2))

# 6. 그룹 연산
grouped_data <- group_by(data, x2)
summarise(grouped_data,
          average_height = mean(x1),
          people = n())
```
## 파이프 연산자 활용
```r
# 기본 사용 예시
data %>% 
  arrange(-x1) %>% 
  filter(x2 == 1)

# 그룹 연산 파이프 체인
data %>% 
  group_by(x2) %>% 
  summarise(x1_mean = mean(x1)) -> new_data

# 체인 중간 결과 확인 (%T>% 사용)
mtcars %>% 
  filter(mpg > 20) %T>% 
  print() %>% 
  summarise(avg_hp = mean(hp))
  ```

 ## 데이터베이스 연동

 ### SQLite 연동 예제

 ```r
 library(DBI)
library(RSQLite)

# 메모리 데이터베이스 연결
con <- dbConnect(RSQLite::SQLite(), ":memory:")
copy_to(con, mtcars, "mtcars_table")

# 데이터 조회 및 분석
db_tbl <- tbl(con, "mtcars_table")
result <- db_tbl %>% 
  filter(mpg > 20) %>% 
  summarise(mean_hp = mean(hp))

show_query(result)  # 생성된 SQL 쿼리 확인
dbDisconnect(con)
```

### MariaDB 연동 예제
```r
library(RMariaDB)

con <- dbConnect(
  RMariaDB::MariaDB(),
  dbname = "daejeon",
  host = "34.64.192.74",
  port = 3306,
  user = "root",
  password = "pakr@0607"
)

# 학생 데이터 분석 예제
student <- dbReadTable(con, "student")
student %>%
  select(kor, mat, eng) %>% 
  summarise(across(everything(), ~ round(mean(.x, na.rm = TRUE), 2)))

dbDisconnect(con)
```

## 실습 문제
### filtering
```r
df <- tibble(
  id = 1:5,
  name = c("대한","민국","만세","영원","무궁"),
  age = c(25,30,35,40,45),
  gender = c("Male","Male","Male","Female","Female")
)

# 30세 이상 필터링 및 내림차순 정렬
filtered <- df %>% 
  filter(age >= 30) %>% 
  arrange(desc(age))
```

### 데이터 변환
```r
wide_data <- data.frame(
  학생 = c("철수", "영희"),
  수학 = c(80,85),
  영어 = c(90,95),
  과학 = c(85,80)
)

# 장형 데이터 변환
long_data <- wide_data %>% 
  pivot_longer(cols = -학생, 
               names_to = "과목", 
               values_to = "점수")

# 과목별 평균 계산
subject_avg <- long_data %>% 
  group_by(과목) %>% 
  summarise(평균_점수 = mean(점수))
```

### 결측치 처리
```r
df <- tibble(
  학생 = c("철수", "영희","민수","지혜"),
  수학 = c(80,NA,75,90),
  영어 = c(85,90,NA,95),
  과학 = c(NA,80,70,85)
)

# 결측치 제거
df_no_na <- df %>% 
  drop_na()            # drop_na() : 열기준 작업
print(df_no_na)

# 선택과목을 중심으로 결측치 제거
df_no_na_specific <- df %>% 
  drop_na(수학)
print(df_no_na_specific)

# 결측치를 0으로 대체
df_replace <- df %>% 
  replace_na(list(
    수학 = 0,
    영어 = 0,
    과학 = 0
  ))
print(df_replace)

# 결측치를 평균으로 대체
df_replace_mean <- df %>% 
  mutate(
    수학 = replace_na(수학, mean(수학, na.rm =T)),
    영어 = replace_na(영어, mean(영어, na.rm =T)),
    과학 = replace_na(과학, mean(과학, na.rm =T))
  )
print(df_replace_mean)

# 결측치를 이전 값으로 대체
df_fill_down <- df %>% 
  fill(수학,영어, .direction = "down")  
print(df_fill_down)


df_fill_up <- df %>% 
  fill(수학, 과학, .direction = "up")
print(df_fill_up)
```

TIP: **%>%** `파이프 연산자`를 사용하면 코드 가독성을 크게 향상시킬 수 있습니다. 데이터 처리 단계를 자연스러운 흐름으로 표현할 수 있어 복잡한 데이터 변환 작업도 쉽게 관리할 수 있습니다