---
title: MEORING CDSS
date: 2025-06-30 1:27:00 +09:00
categories: [CDSS]
tags: [CDSS]
---

# MEORING CDSS (Clinical Decision Support System)

**GitHub Repository**: [https://github.com/ParkYaeseong/CDSS.git](https://github.com/ParkYaeseong/CDSS.git)


## 프로젝트 소개 (About The Project)

**MEORING CDSS**는 의료 데이터 속에 숨겨진 **'의미(Meaning)'**를 찾아내어 암 조기 진단 및 정밀 의료 분야에 혁신을 가져오는 AI 기반 임상 의사 결정 지원 시스템입니다.

본 프로젝트는 환자의 임상 정보, 고해상도 CT 영상, 유전체/단백체 등 다중 오믹스(Multi-omics) 데이터를 유기적으로 통합 분석하여, 의료진이 최적의 진단과 치료 결정을 내릴 수 있도록 지원하는 것을 목표로 합니다. OpenEMR과 같은 기간계 시스템과의 연동을 통해 실제 진료 워크플로우에 통합될 수 있는 확장성 높은 아키텍처를 지향합니다.

> ### **MEORING** (Medical Enhanced Omics Real-time Integrated Navigation Guidance)
>
>   - **M**edical: 의료 현장과 환자 중심의 가치를 담은
>   - **E**nhanced: 최신 AI 기술로 고도화된 분석 능력
>   - **O**mics: 오믹스 데이터를 포함한 다차원 데이터
>   - **R**eal-time: 신속하고 실시간에 가까운 정보 제공
>   - **I**ntegrated: 분절된 데이터를 통합하여 종합적인 인사이트 제공
>   - **N**avigation: 의료진의 복잡한 의사결정을 위한 명확한 길잡이 역할
>   - **G**uidance: 정확한 진단과 치료를 위한 지능형 지원


## 주요 기능 (Core Features)

  - **다중 모달 데이터 통합 대시보드**: 오믹스(유전체, 단백질 등 5종), CT 영상(3D), 임상 데이터를 통합하여 환자 상태를 직관적으로 파악할 수 있는 동적 대시보드를 제공합니다.
  - **AI 기반 정밀 진단 및 예측**:
      - 주요 암(간암, 신장암, 위암 등)에 대한 위험도 분류, 생존율 예측, 치료 효과 예측
      - 5대 암종(유방암, 간암, 위암 등) 분류
      - 설명 가능한 AI (XAI)를 통해 SHAP, Feature Importance 등 예측의 근거를 시각적으로 제시합니다.
  - **CT 영상 자동 분할 및 3D 시각화**: 3D U-Net 기반 AI 모델이 CT 영상 내 종양 및 주요 장기를 자동으로 분할하고, 3D 모델로 시각화하여 부피 측정 등 정밀 분석을 지원합니다.
  - **규칙 기반 약물 상호작용(DDI) 검사**: 공공 데이터 API(DUR)와 연동하여 처방 약물 간의 상호작용, 병용 금기 정보를 실시간으로 확인하여 처방 안전성을 높입니다.
  - **RAG 기반 의료 챗봇**: Google Gemini API와 VectorDB에 저장된 최신 의학 정보를 활용하여, 근거 기반의 질문-답변을 제공하는 AI 챗봇 기능을 제공합니다.
  - **스마트 의료 워크플로우**:
      - 환자 접수 및 예약 관리, AI 간호일지 자동 생성
      - 의료진-환자 간 보안 메시징 및 환자용 모바일 앱(Flutter) 연동
  - **외부 시스템 연동**: OpenEMR(EHR), Orthanc(PACS), LIS 등 기존 병원 정보 시스템과의 유기적인 데이터 연동을 지원합니다.

## 시스템 아키텍처 (System Architecture)

MEORING CDSS는 Django 기반의 강력한 백엔드와 React/Flutter 기반의 유연한 프론트엔드로 구성되며, 각 모듈은 유기적으로 연결되어 정밀 의료 워크플로우를 지원합니다.

```mermaid
graph TD
    subgraph User Interface
        A[React Web App <br> (의료진용)]
        B[Flutter Mobile App <br> (환자용)]
    end

    subgraph Backend API Server
        C[Django REST Framework <br> (Business Logic, API Gateway)]
    end

    subgraph Asynchronous Services
        D[Celery]
        E[Redis <br> (Message Broker)]
    end

    subgraph AI/ML Models
        F[임상 예측 모델 <br> (XGBoost, LightGBM, etc.)]
        G[CT 영상 분할 모델 <br> (3D U-Net)]
        H[오믹스 분석 파이프라인]
        I[생성형 AI <br> (Google Gemini - RAG)]
    end

    subgraph Data Storage
        J[MySQL <br> (메인 데이터베이스)]
        K[VectorDB <br> (RAG용 의료 정보)]
    end

    subgraph External Systems
        L[OpenEMR <br> (EHR/EMR)]
        M[Orthanc <br> (PACS)]
        N[LIS <br> (검사정보시스템)]
        O[공공 데이터 API <br> (DUR, 병원정보 등)]
    end

    A & B -->|REST API (Axios)| C
    C -->|JWT Auth| A & B
    C -->|Create Task| D
    D -->|Get Task| E
    D -->|Process Task| F & G & H & I
    C -->|Store/Retrieve Data| J
    C -->|Query| K
    C -->|API Calls| L & M & N & O
```

1.  **Frontend (React/Flutter)**: 사용자 역할(의사, 간호사, 환자 등)에 따라 맞춤형 UI/UX를 제공하며, 백엔드 API와 비동기 통신(Axios)을 수행합니다.
2.  **Backend API (Django REST Framework)**: 모든 비즈니스 로직, 데이터 처리, AI 모델 오케스트레이션, 외부 시스템 통신을 담당하는 중앙 허브입니다. JWT 기반으로 인증/인가를 관리합니다.
3.  **AI/ML Services (Celery & Redis)**: 대규모 오믹스 데이터 처리, CT 영상 분할, AI 모델 추론 등 시간이 많이 소요되는 작업을 비동기적으로 처리하여 시스템 응답성을 확보합니다.
4.  **Database (MySQL & VectorDB)**: 환자 정보, 진료 기록, 분석 결과 등 정형 데이터는 MySQL에, RAG 챗봇을 위한 비정형 의료 텍스트 데이터는 VectorDB에 저장합니다.
5.  **External Integrations**: OpenEMR, Orthanc(PACS), LIS, 공공 API 등 다양한 외부 시스템과 연동하여 데이터의 확장성과 서비스의 유용성을 극대화합니다.



## 기술 스택 (Tech Stack)

| 구분                  | 기술                                                                                              |
| --------------------- | ------------------------------------------------------------------------------------------------- |
| **Backend** | `Python`, `Django`, `Django REST Framework`, `Celery`, `Gunicorn`                                   |
| **Frontend** | `React.js`, `Flutter`, `JavaScript`, `Dart`, `Axios`, `React Router`                                |
| **AI / ML** | `PyTorch`, `TensorFlow`, `Scikit-learn`, `XGBoost`, `LightGBM`, `SHAP`, `MONAI (3D U-Net)`            |
| **Generative AI** | `Google Gemini API`, `RAG`, `VectorDB`                                                            |
| **Database** | `MySQL`, `PostgreSQL`                                                                             |
| **Medical Standard** | `OpenEMR`, `PACS (Orthanc)`, `DICOM (SimpleITK, DCMTK)`, `FHIR`                                     |
| **Infra & DevOps** | `Docker`, `Nginx`, `Ubuntu`                                                                       |
| **API Docs** | `drf-spectacular (Swagger/ReDoc)`                                                                 |


## 페이지
**(1. 임상데이터 분석 장면)**
![image](assets/cdss/clinic.png)

**(2. Omics 분석 장면)**
![image](assets/cdss/omics.png)

**(3. CT 분석 장면)**
![image](assets/cdss/ct.png)

**(4. CDSS와 flutter 연동)**
![image](assets/cdss/flutter.png)
