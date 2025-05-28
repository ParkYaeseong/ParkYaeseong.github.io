---
title: nnU-Net 사전 훈련 모델(MAMA-MIA)을 이용한 유방암 종양 분할 과정정 및 트러블슈팅
date: 2025-05-26 01:28:00 +09:00
categories: [Project, 암 예후 예측]
tags: [의료 데이터, CDSS]
---

# nnU-Net 사전 훈련 모델(MAMA-MIA)을 이용한 유방암 종양 분할 과정정 및 트러블슈팅

## 목표

이 문서는 사전 훈련된 nnU-Net 모델(MAMA-MIA 프로젝트, Dataset ID: 101)을 사용하여 유방암 DCE-MRI 영상에서 종양 영역을 분할하고, 생성된 마스크를 후속 암 분류 모델에 활용하기까지의 준비 과정과 발생했던 다양한 문제 해결 과정을 기록한 것입니다.

## 1. 초기 환경 설정 및 데이터 준비

프로젝트의 시작은 데이터 준비와 필요한 도구 설치였습니다.

* **DICOM -> NIfTI 변환**: 보유하고 있는 DICOM CT/MRI 데이터를 NIfTI 형식(`.nii.gz`)으로 변환하는 사용자 정의 Python 스크립트(`preprocess_dicom_to_nifti.py`)를 사용했습니다. 이 스크립트는 MONAI 라이브러리를 활용하여 이미지 로딩, Orientation 및 Spacing 조정 등의 전처리도 수행합니다.
* **nnU-Net v2 설치**: 의료 영상 분할에 강력한 프레임워크인 nnU-Net 최신 버전(v2)을 `pip install nnunetv2` 명령을 통해 설치했습니다.
* **nnU-Net 환경 변수 설정**: nnU-Net이 올바르게 작동하기 위해 필요한 환경 변수들을 설정했습니다. 특히 Windows 환경에서 작업했으므로, CMD에서는 `set` 명령어를, 영구적으로는 시스템 속성을 통해 다음 변수들을 지정했습니다:
    * `nnUNet_results`: 학습된 모델 및 다운로드한 사전 훈련 모델이 저장될 경로.
    * `nnUNet_raw`: 원본 데이터셋(주로 모델 훈련 시 사용) 경로.
    * `nnUNet_preprocessed`: nnU-Net이 내부적으로 전처리한 데이터 저장 경로.

## 2. MAMA-MIA (Dataset 101) 사전 훈련 모델 준비

유방암 종양 분할을 위해 MAMA-MIA 프로젝트에서 제공하는 사전 훈련된 nnU-Net 모델을 사용하기로 결정했습니다.

* **모델 다운로드**: Synapse 저장소 (ID: `syn60868042`)의 `nnUNet_pretrained_weights` 폴더에서 `full_image_dce_mri_tumor_segmentation.zip` 파일을 다운로드했습니다.
* **올바른 폴더 구조로 배치**: 다운로드한 압축 파일의 내용물을 `nnUNet_results` 환경 변수에 지정된 경로 하위에 다음과 같은 nnU-Net 표준 구조에 맞게 배치하는 것이 중요했습니다.
    * `<nnUNet_results 경로>\Dataset101_BreastCancerTumor\nnUNetTrainer__nnUNetPlans__3d_fullres\`
    * 이 경로 안에 `dataset.json`, `plans.json` 및 각 fold의 모델 가중치 파일(`fold_0`, `fold_1` 등)이 위치해야 합니다. 초기에는 이 구조가 정확하지 않아 `FileNotFoundError`가 발생했습니다.
* **`dataset.json` 내용 확인**: 모델과 함께 제공된 `dataset.json` 파일을 통해 모델이 단일 채널("T1") 입력을 예상하고, 출력 마스크에서 배경은 0, 종양은 1로 레이블링됨을 확인했습니다.

## 3. nnU-Net 입력용 NIfTI 이미지 준비 및 트러블슈팅

실제 `nnUNetv2_predict` 명령어를 실행하기 전에 입력 데이터와 관련된 몇 가지 문제가 있었습니다.

### 3.1. "0 cases found" 오류 해결

* **원인**: `nnUNetv2_predict` 명령어의 `-i` 옵션으로 지정된 입력 폴더에 nnU-Net이 직접 읽을 수 있는 `.nii.gz` 파일이 없고, 실제 NIfTI 파일들이 다른 압축 파일 내부에 중첩되어 있었습니다. (예: `TCGA-AO-A0J9` 폴더 내 "압축된 보관 파일"들 안에 실제 `.nii.gz` 파일 존재)
* **해결**: Python 스크립트를 작성하여 이중으로 압축된 파일들을 자동으로 해제하고, 실제 `.nii.gz` 파일들만 추출하여 별도의 폴더(`nifti_input_for_nnunet_extracted`)에 플랫(flat)하게 저장했습니다.
    ```python
    # 자동 압축 해제 스크립트 주요 로직 (예시)
    import os
    import shutil
    import zipfile # 실제로는 다양한 압축 형식 지원 필요 가능성
    from tqdm import tqdm

    SOURCE_BASE_DIR = r"C:\Users\사용자\Desktop\modeling\preprocessed_nifti_data"
    EXTRACTED_NIFTIS_OUTPUT_DIR = r"C:\Users\사용자\Desktop\modeling\nifti_input_for_nnunet_extracted"
    os.makedirs(EXTRACTED_NIFTIS_OUTPUT_DIR, exist_ok=True)

    # SOURCE_BASE_DIR 내 환자 폴더 순회
    #   각 환자 폴더 내 압축 파일(들) 순회
    #     압축 파일 형식에 맞게 해제 (예: zipfile 사용)
    #     압축 해제된 내용물 중 .nii.gz 파일만 EXTRACTED_NIFTIS_OUTPUT_DIR로 복사/이동
    #       (파일명 중복 방지를 위한 고유 이름 생성 로직 포함)
    ```

### 3.2. `IndexError: list index out of range` 오류 해결

* **원인**: 입력 NIfTI 파일들의 이름이 nnU-Net이 내부적으로 케이스 식별자(case identifier)를 파싱할 때 예상하는 규칙(특히 `_XXXX.nii.gz` 접미사 관련 길이)과 맞지 않았습니다.
* **해결**: 입력 폴더 내 모든 `.nii.gz` 파일명 끝에 `_0000`을 추가하여 `*식별자*_0000.nii.gz` 형태로 변경하는 Python 스크립트를 사용했습니다. MAMA-MIA 모델은 단일 채널 입력을 받으므로 `_0000` 접미사가 표준적입니다.
    ```python
    # 파일명 변경 스크립트 주요 로직 (예시)
    import os
    input_folder = r"C:\Users\사용자\Desktop\modeling\nifti_input_for_nnunet_extracted"
    # ... (os.listdir, os.path.splitext, os.rename 사용) ...
    # filename_base + "_0000" + ".nii.gz" 형태로 변경
    ```

## 4. `nnUNetv2_predict` 실행 및 트러블슈팅 과정

데이터 준비 후, 실제 예측 과정에서도 여러 문제에 직면했습니다.

### 4.1. `FileNotFoundError` (모델 파일 관련)

* **`dataset.json` 경로 오류**: 처음에는 모델 파일들이 `nnUNet_results` 내의 `Dataset101_BreastCancerTumor` 폴더 바로 아래에 위치하여 발생했습니다. nnU-Net은 특정 트레이너/플랜 이름으로 된 하위 폴더(예: `nnUNetTrainer__nnUNetPlans__3d_fullres`) 내에서 `dataset.json`을 찾기 때문에, 이 구조에 맞게 파일을 이동하여 해결했습니다.
* **`fold_all/checkpoint_final.pth` 경로 오류**: `-f all` (모든 fold 앙상블) 옵션 사용 시 이 경로를 찾는 오류가 발생했습니다. 이는 다운로드한 MAMA-MIA 모델 패키지에 `fold_all`이 없거나, `-f all` 처리 방식이 특정 파일 구조를 요구하기 때문일 수 있습니다. 우선 `-f 0` (단일 fold)으로 테스트하여 개별 fold 모델은 정상 작동함을 확인했습니다.

### 4.2. `RuntimeError: Cannot access accelerator device when none is available.`

* **원인**: Anaconda 환경에 설치된 PyTorch가 CPU 전용 버전(`torch_version+cpu`)이어서 CUDA를 사용할 수 없었습니다.
* **해결**: 기존 CPU 버전 PyTorch를 제거하고, PyTorch 공식 웹사이트에서 CUDA 버전(예: 11.8)을 지원하는 PyTorch 빌드를 Anaconda 명령어로 재설치했습니다.
    ```bash
    # 예시 (Anaconda Prompt, 해당 환경 활성화 후)
    # pip uninstall torch torchvision torchaudio
    # conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
    ```
    이후 `torch.cuda.is_available()`이 `True`로 반환됨을 확인했습니다.

### 4.3. `numpy.core._exceptions._ArrayMemoryError: Unable to allocate ... GiB for an array ...` (시스템 RAM 부족)

* **원인**: 특정 입력 이미지가 nnU-Net의 전처리 단계(특히 리샘플링)에서 매우 큰 배열(예: 15.3 GiB)을 생성하려다 시스템 RAM 부족으로 오류가 발생했습니다. 이는 GPU VRAM 부족과는 다른 문제입니다.
* **시도 1 (`-device cpu`)**: CPU로 강제 실행 시 이 오류는 피할 수 있었으나, 처리 속도가 매우 느려 다른 방법을 모색했습니다.
* **시도 2 (`--disable_tta`)**: Test Time Augmentation을 비활성화했으나, 이는 주로 VRAM 사용량에 영향을 미치므로 RAM 부족 문제에는 큰 도움이 되지 않았습니다.
* **시도 3 (프로세스 수 제한 - 잘못된 인자명)**: `-num_processes_preprocessing 1 -num_processes_segmentation_export 1` 옵션을 사용했으나, `unrecognized arguments` 오류가 발생했습니다.
* **시도 4 (프로세스 수 제한 - 올바른 인자명)**: `usage` 메시지를 참고하여 `-npp 1 -nps 1`로 수정했으나, 여전히 일부 매우 큰 이미지에서 RAM 부족 오류가 재현되었습니다.
* **최종 해결 방안 모색**: "사전 검사(Pre-flight Check)" Python 스크립트를 작성하여, nnU-Net 실행 전에 각 이미지의 리샘플링 후 예상 메모리 사용량을 계산하고, 설정한 임계값을 초과하는 이미지는 별도로 분류하거나 예측 대상에서 제외하는 방식을 고안했습니다.
    ```python
    # 사전 검사 스크립트 주요 로직 (예시)
    import os
    import shutil
    import json
    import SimpleITK as sitk
    import numpy as np
    from tqdm import tqdm

    # MODEL_PLANS_JSON_PATH 에서 target_spacing 읽기
    # SOURCE_NIFTI_DIR 에서 각 이미지 로드 (SimpleITK)
    #   original_size, original_spacing 가져오기
    #   target_spacing 으로 리샘플링 시 new_shape 계산
    #   new_shape 과 데이터 타입(float64, 8 bytes)으로 예상 메모리 계산
    #   예상 메모리가 MEMORY_THRESHOLD_GIB 초과 시 "문제 파일"로 분류, 아니면 SAFE_NIFTI_OUTPUT_DIR로 복사
    ```

### 4.4. 최종 예측 명령어 (RAM 문제 해결 후, 안전한 파일 대상)

사전 검사 스크립트를 통해 걸러진, RAM 문제를 일으키지 않을 것으로 예상되는 이미지들만 포함된 폴더를 대상으로 다음 명령어를 실행합니다.

```batch
nnUNetv2_predict -i C:\Users\사용자\Desktop\modeling\nifti_input_safe_for_nnunet ^
                 -o C:\Users\사용자\Desktop\modeling\MAMA_MIA_tumor_masks_output_fold0 ^
                 -d 101 ^
                 -c 3d_fullres ^
                 -f 0 ^
                 -npp 1 ^
                 -nps 1 ^
                 --disable_tta ^
                 --save_probabilities
```
(위 명령어는 fold_0에 대한 예시이며, 필요에 따라 -f all 또는 다른 fold로 변경하여 실행할 수 있습니다. 단, -f all 사용 시 fold_all/checkpoint_final.pth 관련 문제가 없는지 확인 필요)

## 5. 생성된 마스크 활용 (다음 단계)
이렇게 성공적으로 생성된 세그멘테이션 마스크 파일들은 이제 원래 목표였던 암 분류 모델(ct_classification_train_with_xai_using_nifti.py)에 활용될 준비가 되었습니다.

Manifest 파일 업데이트: 원본 이미지 경로와 함께 생성된 마스크 파일 경로를 포함하도록 preprocessed_manifest.csv 파일을 업데이트합니다.
분류 모델 스크립트 수정:
데이터 로딩 시 이미지와 마스크를 함께 로드합니다.
MONAI Transform 파이프라인에서 마스크에 대해서도 공간적 변환(Resize, Augmentation 등)을 이미지와 일치시키되, 마스크에는 mode='nearest' 보간법을 사용합니다.
로드된 마스크를 활용하여 이미지에서 ROI(관심 영역, 예: 종양)만 추출하거나 배경을 제거하는 등의 방식으로 모델 입력 데이터를 가공합니다. (예: masked_image = image * (mask > 0).float())

## 6. 결론 및 교훈
nnU-Net과 같은 강력한 프레임워크를 사용하더라도, 실제 데이터와 환경에서는 다양한 문제에 직면할 수 있습니다. 이 과정을 통해 다음과 같은 교훈을 얻을 수 있었습니다:

환경 변수 설정의 중요성: nnU-Net은 특정 환경 변수(nnUNet_results 등)를 기준으로 작동하므로, 정확한 설정이 필수적입니다.
사전 훈련 모델의 폴더 구조 이해: 모델 제공자가 의도한 폴더 구조를 정확히 따라야 FileNotFoundError를 피할 수 있습니다.
입력 데이터의 중요성: 파일명 규칙, 실제 파일 접근성(압축 해제 여부) 등이 예측 성공에 큰 영향을 미칩니다.
에러 메시지 분석: 발생하는 오류 메시지를 자세히 읽고 분석하는 것이 단계적인 문제 해결의 핵심입니다.
시스템 자원 한계 인지: GPU VRAM뿐만 아니라 시스템 RAM 또한 대용량 3D 의료 영상 처리 시 병목 지점이 될 수 있으며, 이에 대한 대응 전략(프로세스 수 조절, 사전 검사 등)이 필요합니다.
이러한 트러블슈팅 과정을 통해 얻은 경험이 다른 분들에게도 도움이 되기를 바랍니다.