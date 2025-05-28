---
title: 의료 AI 분할 정복기 Zenodo와 nnU-Net으로 사전 훈련 모델 활용하기
date: 2025-05-28 01:28:00 +09:00
categories: [Project, 암 예후 예측]
tags: [의료 데이터, CDSS]
---

# 의료 AI 분할 정복기: Zenodo와 nnU-Net으로 사전 훈련 모델 활용하기

의료 영상에서 특정 장기나 병변을 정확하게 분할하는 것은 정밀 진단, 치료 계획 수립, 그리고 의학 연구에 있어 매우 중요한 단계입니다. 하지만 고품질의 분할 모델을 처음부터 개발하고 훈련하는 것은 많은 시간과 데이터, 그리고 전문 지식을 필요로 합니다. 다행히도, 이미 잘 훈련된 모델들을 활용할 수 있는 훌륭한 공개 리소스들이 존재합니다.

이 글에서는 NCI Imaging Data Commons (IDC)의 AIMI Annotations 프로젝트가 Zenodo를 통해 제공하는 사전 훈련된 분할 모델들과, 이러한 모델들의 기반이 되는 강력한 프레임워크인 nnU-Net에 대해 자세히 알아보고자 합니다. 특히 간(liver), 신장(kidney), 폐(lung) 등 주요 장기의 분할 모델을 중심으로 살펴보겠습니다.

## 1. AIMI Annotations Zenodo Record: 사전 훈련된 분할 모델의 보고 [cite: 1]

최근 NCI Imaging Data Commons (IDC)의 일부로 "Image segmentations produced by BAMF under the AIMI Annotations initiative"라는 제목의 데이터셋 및 모델 가중치가 Zenodo를 통해 공개되었습니다[cite: 1]. 이는 공개적으로 사용 가능한 암 영상 데이터에 AI 기반 분석을 통해 생성된 분할(annotation) 데이터를 확충하려는 노력의 일환입니다[cite: 1].

### 주요 특징 및 제공 내용:

* **nnU-Net 기반 모델 사용**: 다양한 영상 분할 작업을 위해 공개 데이터셋으로 훈련된 여러 nnU-Net 기반 모델을 사용해 IDC 컬렉션의 분할 결과를 생성했습니다[cite: 1].
* **검증 프로세스**: 생성된 AI 분할 결과 중 약 10%에 대해 영상의학과 전문의가 Likert 척도(1~5점)를 사용하여 품질을 평가하고, '매우 동의함(5점)'이 아닌 경우 분할을 직접 수정했습니다[cite: 1].
* **제공 파일 (각 작업별 .zip 파일)**[cite: 1]:
    * `ai-segmentations-dcm`: AI 모델 예측 결과 (DICOM-SEG 형식).
    * `qa-segmentations-dcm`: AI 예측 기반으로 수동으로 수정된 분할 결과 (약 10%).
    * `qa-results.csv`: 분할 관련 메타데이터, IDC 케이스 정보, 검토자의 평가 점수 및 코멘트.
* **사전 훈련된 모델 가중치**: 각 분할 작업에 사용된 nnU-Net 모델의 가중치(weights)에 대한 링크를 제공하여, 사용자가 직접 해당 모델을 다운로드하여 자신의 데이터에 적용해볼 수 있도록 합니다[cite: 1].

### 주목할 만한 사전 훈련 모델 (Zenodo 페이지 내용 기반)[cite: 1]:

Zenodo 기록의 "File Overview" 섹션에는 다양한 장기 및 병변에 대한 분할 모델과 관련 정보가 포함되어 있습니다. 특히 우리가 관심 있는 장기에 대해서는 다음과 같은 모델들이 언급됩니다:

* **`kidney-ct.zip`**:
    * 분할 대상: 조영 증강 CT 스캔에서의 신장(Kidney), 종양(Tumor), 낭종(Cysts)[cite: 1].
    * 대상 IDC 컬렉션: TCGA-KIRC, TCGA-KIRP, TCGA-KICH, CPTAC-CCRCC[cite: 1].
    * **제공 링크**: 모델 가중치, GitHub[cite: 1].
* **`liver-ct.zip`**:
    * 분할 대상: CT 스캔에서의 간(Liver)[cite: 1].
    * 대상 IDC 컬렉션: TCGA-LIHC[cite: 1].
    * **제공 링크**: 모델 가중치, GitHub[cite: 1]. (이전에 `Task773_Liver`로 사용해 보려 했던 모델과 관련 있을 가능성이 높습니다.)
* **`liver2-ct.zip`**:
    * 분할 대상: CT 스캔에서의 간(Liver) 및 병변(Lesions)[cite: 1].
    * 대상 IDC 컬렉션: HCC-TACE-SEG, COLORECTAL-LIVER-METASTASES[cite: 1].
    * **제공 링크**: 모델 가중치, GitHub[cite: 1].
* **`lung-ct.zip`**:
    * 분할 대상: CT 스캔에서의 폐(Lung) 및 결절(Nodules, 3mm-30mm)[cite: 1].
    * 대상 IDC 컬렉션: Anti-PD-1-Lung, LUNG-PET-CT-Dx, NSCLC Radiogenomics, RIDER Lung PET-CT, TCGA-LUAD, TCGA-LUSC[cite: 1].
    * **제공 링크**: 모델 가중치 1, 모델 가중치 2, GitHub[cite: 1].
* **`lung2-ct.zip` (개선된 버전)**:
    * 분할 대상: CT 스캔에서의 폐(Lung) 및 결절(Nodules, 3mm-30mm)[cite: 1].
    * 대상 IDC 컬렉션: QIN-LUNG-CT, SPIE-AAPM Lung CT Challenge[cite: 1].
    * **제공 링크**: 모델 가중치, GitHub[cite: 1].

이 외에도 뇌종양(brain-mr), 유방(breast-mr, breast-fdg-pet-ct), 전립선(prostate-mr) 등 다양한 부위에 대한 모델들이 제공됩니다[cite: 1]. 각 항목의 "model weights" 링크를 통해 nnU-Net 모델 파일을 다운로드할 수 있습니다.

## 2. nnU-Net 프레임워크: "설정 없는(No New U-Net)" 분할 솔루션 [cite: 2]

AIMI Annotations에서 사용된 모델들의 기반이 되는 nnU-Net은 의료 영상 분할을 위한 매우 강력하고 유연한 프레임워크입니다.

### nnU-Net 주요 특징 및 장점[cite: 2]:

* **자동 구성**: 새로운 데이터셋이 주어지면, nnU-Net은 해당 데이터의 특성(이미지 차원, 모달리티, 크기, 간격, 클래스 비율 등)을 자동으로 분석하고 최적화된 U-Net 기반 파이프라인을 구성합니다[cite: 2]. 사용자의 전문적인 개입이 거의 필요 없습니다.
* **다양한 데이터 처리**: 2D 및 3D 이미지, 임의의 입력 채널/모달리티, 다양한 복셀 간격 및 이방성(anisotropies)을 처리할 수 있으며, 클래스 불균형에도 강인합니다[cite: 2].
* **성과 입증**: 수많은 의료 영상 분할 챌린지에서 최상위권의 성능을 기록하며 그 효과를 입증했습니다[cite: 2]. MICCAI 2020년 및 2021년 챌린지 우승팀 다수가 nnU-Net을 기반으로 솔루션을 개발했습니다[cite: 2].
* **지도 학습 기반**: 고품질의 분할 결과를 얻기 위해서는 해당 작업에 대한 훈련 데이터(이미지 및 레이블 마스크)가 필요합니다[cite: 2].
* **파이프라인 구성 방식**[cite: 2]:
    * **고정 매개변수**: 손실 함수, 대부분의 데이터 증강 전략, 학습률 등은 견고성이 입증된 값으로 고정됩니다.
    * **규칙 기반 매개변수**: 데이터셋 특성(핑거프린트)을 분석하여 네트워크 토폴로지, 패치 크기, 배치 크기 등을 휴리스틱 규칙에 따라 조정합니다.
    * **경험적 매개변수**: 최적의 U-Net 구성(2D, 3D full resolution, 3D cascade 등) 선택 및 후처리 전략 최적화 등은 실험적으로 결정됩니다.
* **최신 Residual Encoder 프리셋**: 최근에는 특히 대규모 데이터셋에서 성능 향상을 가져오는 새로운 Residual Encoder UNet 프리셋(nnU-Net ResEnc M, L, XL)이 도입되었습니다[cite: 2].

### nnU-Net 시작하기 (README 기반)[cite: 2]:

nnU-Net GitHub 저장소의 README는 시작에 필요한 모든 정보를 제공합니다:
1.  **설치 지침 (Installation instructions)**
2.  **데이터셋 변환 (Dataset conversion)**: 자신만의 데이터로 모델을 훈련시키기 위한 형식 변환 방법.
3.  **사용 지침 (Usage instructions)**: 모델 훈련 및 예측 실행 방법.

## 3. AIMI 모델과 nnU-Net의 시너지 활용 방안

Zenodo의 AIMI Annotations 기록에서 제공하는 "model weights"는 nnU-Net 프레임워크와 호환되는 사전 훈련된 모델들입니다. 이를 활용하는 일반적인 과정은 다음과 같습니다:

1.  **Zenodo에서 원하는 모델 가중치 다운로드**: 관심 있는 장기(예: `liver-ct.zip`, `kidney-ct.zip`, `lung-ct.zip`)의 "model weights" 링크를 통해 모델 압축 파일을 다운로드합니다.
2.  **nnU-Net 환경 변수 설정**: `nnUNet_results` 환경 변수가 올바르게 설정되어 있는지 확인합니다. 이 경로 하위에 다운로드한 모델이 설치됩니다.
3.  **모델 설치**: 다운로드한 모델 압축 파일(`.zip`)을 `nnUNetv2_install_pretrained_model_from_zip` 명령어를 사용하여 nnU-Net에 설치합니다.
    ```bash
    nnUNetv2_install_pretrained_model_from_zip <다운로드한_모델_zip파일_경로>
    ```
    이 명령은 모델 파일들을 `nnUNet_results` 폴더 내에 적절한 Dataset/Task ID 이름으로 된 하위 폴더 구조에 맞게 배치해줍니다. (예: `nnUNet_results/DatasetXXX_OrganName/Trainer__Plans__Config/`)
4.  **예측 실행**: 설치된 모델을 사용하여 자신의 데이터에 대해 분할을 수행합니다. `nnUNetv2_predict` 명령어를 사용하며, `-d` 옵션에 해당 모델의 Dataset ID 또는 Task ID를 지정합니다.
    ```bash
    nnUNetv2_predict -i <입력_NIfTI_폴더> -o <출력_마스크_폴더> -d <설치된_모델의_DatasetID> -c 3d_fullres -f all [기타_옵션]
    ```

## 결론

NCI IDC AIMI Annotations와 nnU-Net 프레임워크는 의료 영상 AI 분할 연구 및 적용에 있어 매우 귀중한 자원입니다. AIMI Annotations를 통해 특정 장기(간, 신장, 폐 등)에 대해 이미 훈련되고 검증까지 거친 nnU-Net 기반 모델 가중치를 쉽게 얻을 수 있으며, nnU-Net의 강력한 예측 기능을 활용하여 자신의 데이터에 빠르게 적용해 볼 수 있습니다.

물론, 실제 사용 과정에서는 데이터 형식, 경로 설정, 시스템 환경(RAM, GPU, CUDA), 라이브러리 버전 등 다양한 기술적 문제에 직면할 수 있습니다. 하지만 이러한 문제들을 단계적으로 해결해 나가는 과정 자체가 중요한 학습 경험이 될 것입니다. 이 글에서 공유된 정보와 문제 해결 과정들이 여러분의 연구에 도움이 되기를 바랍니다!