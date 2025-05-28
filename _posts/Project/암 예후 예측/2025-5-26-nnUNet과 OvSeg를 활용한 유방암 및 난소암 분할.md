---
title: 의료 영상 분할 nnU-Net과 OvSeg를 활용한 유방암 및 난소암 분할
date: 2025-05-27 01:28:00 +09:00
categories: [Project, 암 예후 예측]
tags: [의료 데이터, CDSS]
---

# 의료 영상 분할 여정: nnU-Net과 OvSeg를 활용한 유방암 및 난소암 분할

## 서론

본 문서는 사전 훈련된 딥러닝 모델을 활용하여 유방암 및 난소암 의료 영상 분할을 시도하고, 그 과정에서 겪었던 다양한 기술적 문제들과 해결 과정을 기록한 여정입니다. 처음에는 nnU-Net 프레임워크를 사용하여 유방암 종양 분할을 시도했고, 이후에는 CT 기반 난소암 분할을 위해 `ovseg` 라이브러리를 탐색했습니다. 이 기록이 비슷한 작업을 수행하는 분들께 작은 도움이 되기를 바랍니다.

---

## 1부: nnU-Net을 이용한 유방암 종양 분할 (MAMA-MIA 모델)

첫 번째 목표는 MAMA-MIA 프로젝트(Dataset ID: 101)에서 제공하는 사전 훈련된 nnU-Net 모델을 사용하여 유방암 DCE-MRI 영상에서 종양 영역을 분할하는 것이었습니다.

### 1.1. 목표 및 초기 설정

* **분할 대상**: 유방암 DCE-MRI 영상 내 종양 영역
* **사용 모델**: MAMA-MIA (Dataset ID: 101) nnU-Net 사전 훈련 모델
* **초기 환경 준비**:
    * 보유한 DICOM 데이터를 NIfTI 형식(`.nii.gz`)으로 변환했습니다. 이 과정은 사용자 정의 Python 스크립트(`preprocess_dicom_to_nifti.py`)를 사용했으며, MONAI 라이브러리를 통해 이미지 로딩, Orientation 조정 (RAS), Spacing 조정 (`(1.5, 1.5, 2.0)`) 등의 기본 전처리를 수행했습니다.
    * nnU-Net v2를 `pip install nnunetv2` 명령으로 설치했습니다.
    * nnU-Net 작동에 필수적인 환경 변수(`nnUNet_results`, `nnUNet_raw`, `nnUNet_preprocessed`)를 설정했습니다. Windows 환경이었기에 CMD에서는 `set` 명령어를 사용하고, 장기적으로는 시스템 속성의 환경 변수 편집기를 통해 영구 설정했습니다.
        * `nnUNet_results`: 모델 가중치 및 예측 결과 저장 경로
        * `nnUNet_raw`: 원본 데이터셋 경로 (주로 훈련 시)
        * `nnUNet_preprocessed`: nnU-Net 내부 전처리 데이터 저장 경로

### 1.2. MAMA-MIA 모델 준비 및 관련 트러블슈팅

모델을 다운로드하고 nnU-Net이 인식할 수 있도록 배치하는 과정에서 몇 가지 문제가 있었습니다.

* **모델 다운로드**: MAMA-MIA 프로젝트의 Synapse 저장소에서 모델 가중치 파일(`full_image_dce_mri_tumor_segmentation.zip`)을 다운로드했습니다.
* **폴더 구조 문제 및 해결**:
    * **초기 오류**: `FileNotFoundError` (예: `dataset.json` 또는 `checkpoint_final.pth` 찾을 수 없음) 발생.
    * **원인**: 다운로드한 모델 파일들을 `nnUNet_results` 폴더 내에 nnU-Net이 예상하는 정확한 하위 폴더 구조로 배치하지 않았기 때문입니다. nnU-Net은 특정 트레이너 및 플랜 이름으로 구성된 경로(예: `Dataset101_BreastCancerTumor/nnUNetTrainer__nnUNetPlans__3d_fullres/`)에서 모델 관련 파일을 찾습니다.
    * **해결 과정**:
        1.  `nnUNet_results` 폴더 바로 아래에 `Dataset101_BreastCancerTumor` (또는 모델 ID 101에 해당하는 정확한 폴더명) 폴더를 생성했습니다.
        2.  그 안에 다시 오류 메시지에 명시된 대로 `nnUNetTrainer__nnUNetPlans__3d_fullres` 라는 하위 폴더를 만들었습니다.
        3.  압축 해제한 모델 파일들(각 `fold_X` 폴더, `dataset.json`, `plans.json` 등)을 이 가장 안쪽 폴더로 이동시켰습니다.
* **`dataset.json` 확인**: 모델과 함께 제공된 `dataset.json` 파일을 통해 모델이 단일 채널("T1" 영상) 입력을 사용하며, 출력 마스크에서 배경은 0, 종양은 1로 레이블링된다는 것을 확인했습니다.

### 1.3. nnU-Net 입력용 NIfTI 이미지 준비 및 트러블슈팅

모델 준비 후, 실제 NIfTI 이미지를 입력으로 사용하는 과정에서도 문제가 발생했습니다.

#### 1.3.1. "0 cases found" 오류

* **원인**: `nnUNetv2_predict` 명령어의 `-i` 옵션으로 지정된 입력 폴더에 nnU-Net이 직접 읽을 수 있는 `.nii.gz` 파일이 없었습니다. 실제 NIfTI 파일들이 다른 압축 파일 내부에 한 번 더 압축되어 있었습니다.
* **해결**: Python 스크립트 (`unzip_nifti_script.py`)를 작성하여, 환자별 폴더 내의 각 압축 파일을 자동으로 해제하고, 그 안에 있는 실제 `.nii.gz` 파일들만 추출하여 별도의 폴더 (`nifti_input_for_nnunet_extracted`)에 플랫(flat)하게 저장했습니다. 이 스크립트는 `zipfile` 모듈을 사용하고, 파일명 중복을 피하기 위한 로직을 포함했습니다.

#### 1.3.2. `IndexError: list index out of range` 오류

* **원인**: "0 cases found" 오류 해결 후, nnU-Net이 파일명에서 케이스 식별자(case identifier)를 파싱하는 과정에서 이 오류가 발생했습니다. 이는 파일명이 nnU-Net이 예상하는 특정 패턴(특히 `_XXXX.nii.gz` 형태의 접미사 및 그 길이)과 일치하지 않았기 때문입니다.
* **해결**: `nifti_input_for_nnunet_extracted` 폴더 내의 모든 `.nii.gz` 파일명 끝에 `_0000`을 추가하여 (예: `기존파일명_0000.nii.gz`) nnU-Net의 단일 모달리티 파일명 규칙을 따르도록 Python 스크립트로 일괄 변경했습니다.

### 1.4. `nnUNetv2_predict` 실행 및 추가 트러블슈팅

데이터 준비가 완료된 후에도 예측 실행 과정에서 여러 오류를 만났습니다.

#### 1.4.1. `RuntimeError: Cannot access accelerator device when none is available.`

* **원인**: Anaconda 환경에 설치된 PyTorch가 CPU 전용 버전(`torch_version+cpu`)이어서 CUDA GPU를 인식하지 못했습니다.
* **해결**:
    1.  `gpu.py` 테스트 스크립트로 `torch.cuda.is_available()`이 `False`임을 확인했습니다.
    2.  기존 CPU 버전 PyTorch를 제거 (`pip uninstall torch ...`).
    3.  PyTorch 공식 웹사이트에서 CUDA 지원 버전(예: CUDA 11.8)의 PyTorch 설치 명령어를 확인하여 Anaconda 환경에 재설치했습니다 (`conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia`).
    4.  재설치 후 `torch.cuda.is_available()`이 `True`로 변경됨을 확인했습니다.

#### 1.4.2. `numpy.core._exceptions._ArrayMemoryError: Unable to allocate ... GiB for an array ...` (시스템 RAM 부족)

* **원인**: GPU 사용이 가능해진 후에도, 특정 입력 이미지를 전처리(특히 리샘플링)하는 과정에서 매우 큰 중간 배열(예: 15.3 GiB의 `float64` 배열)을 생성하려다 시스템 RAM 부족으로 오류가 발생했습니다.
* **시도된 해결책 및 과정**:
    1.  `-device cpu` 옵션: 성공했으나, 처리 속도가 매우 느려 다른 방법을 찾기로 했습니다.
    2.  `--disable_tta` 옵션: GPU VRAM 절약에는 도움이 되지만, 시스템 RAM 문제에는 직접적인 효과가 없었습니다.
    3.  프로세스 수 제한 시도 (잘못된 인자명): `-num_processes_preprocessing 1 -num_processes_segmentation_export 1` 옵션을 사용했으나, `unrecognized arguments` 오류가 발생했습니다.
    4.  프로세스 수 제한 (올바른 인자명): `usage` 메시지를 참고하여 `-npp 1 -nps 1`로 수정하여 실행했으나, 여전히 일부 매우 큰 이미지에서 RAM 부족 오류가 재현되었습니다.
    5.  **"사전 검사(Pre-flight Check)" 스크립트 도입**: RAM 사용량 초과가 예상되는 이미지를 `nnUNetv2_predict` 실행 전에 미리 식별하여 제외/분리하는 Python 스크립트 (`check_images.py`)를 작성하고 실행했습니다.
        * 이 스크립트는 MAMA-MIA 모델의 `plans.json` 파일에서 목표 리샘플링 간격(예: `[1.0, 1.0, 1.0]`)을 읽어옵니다.
        * 각 입력 NIfTI 파일의 원본 크기/간격을 `SimpleITK`로 읽어, 목표 간격으로 리샘플링 시 예상되는 메모리 크기를 계산합니다. (이때, 오류 로그를 참고하여 복셀당 8바이트(`float64`) 기준으로 계산하고, `np.prod`의 정수 오버플로우를 방지하기 위해 `dtype=np.longlong` 사용 등 계산 정확도를 높였습니다.)
        * 설정한 메모리 임계값(예: 10GiB)을 초과하는 파일은 "문제 파일"로 분류하고, "안전한" 파일들만 새 폴더(`nifti_input_safe_for_nnunet_v2`)로 복사했습니다.
        * 스크립트 실행 결과, 1724개 중 1514개의 "안전한" 파일이 복사되었고, 210개 파일(일부는 예상 메모리 80GiB 이상)이 제외되었습니다.

### 1.5. 최종 `nnUNetv2_predict` 명령어 (안전한 파일 대상)

"사전 검사"를 통과한 파일들이 담긴 폴더를 대상으로 다음 명령어를 사용하여 예측을 성공적으로 진행했습니다.

```batch
nnUNetv2_predict -i C:\Users\사용자ID\Desktop\modeling\nifti_input_safe_for_nnunet_v2 ^
                 -o C:\Users\사용자ID\Desktop\modeling\MAMA_MIA_tumor_masks_output_fold0_safe_v2 ^
                 -d 101 ^
                 -c 3d_fullres ^
                 -f 0 ^
                 -npp 1 ^
                 -nps 1 ^
                 --disable_tta ^
                 --save_probabilities