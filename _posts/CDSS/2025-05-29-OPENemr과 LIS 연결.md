---
title: PACS 서버 구축 가이드
date: 2025-04-29 10:27:00 +09:00
categories: [CDSS]
tags: [CDSS, Orthanc, PACS server]
---

# 차세대 지능형 의료지원 시스템(CDSS) 구축 여정: OpenEMR, Django, 그리고 오픈소스 연동의 모든 것 (v2)

안녕하세요! 현재 진행 중인 차세대 지능형 의료지원 시스템(CDSS) 구축 프로젝트의 여정과 기술적 선택들을 지속적으로 공유하고자 합니다. 이 시스템은 **OCS(처방 전달 시스템)를 핵심 축**으로 하며, 의료진의 의사 결정을 지원하고 AI를 통해 진단 정확도를 높이는 것을 목표로 합니다. 주요 기술 스택으로는 OpenEMR, Django를 사용하며, 다양한 오픈소스 시스템과의 연동을 적극적으로 활용하고 있습니다.

## 프로젝트 개요 및 초기 설정 (Phase 1 & 2)

본 프로젝트의 핵심은 Django를 사용하여 OCS(처방 전달 시스템) 백엔드 로직을 구축하는 것입니다. 이 OCS 모듈은 사용자의 처방 입력을 받아 LIS, OpenEMR, PACS 등 기간계 시스템과 연동되고, CDW를 거쳐 AI 분석으로 이어지는 전체 워크플로우의 시작점이 됩니다. 사용자 인터페이스(프론트엔드)는 React(웹)와 Flutter(모바일) 사용을 기본으로 고려하고 있습니다.

## 핵심 병원 시스템 연동: 도전과 해결 (Phase 3)

CDSS가 제 기능을 하려면 기존 병원 정보 시스템과의 원활한 연동이 필수적입니다. 이 과정에서 몇 가지 도전 과제에 직면했고, 이를 해결하기 위한 접근 방식을 정리해 보았습니다.

### 1. OpenEMR (EMR) 연동 및 기능 확장

OpenEMR은 우리 시스템의 중심 EMR 역할을 합니다. 단순 데이터 연동을 넘어, OpenEMR 자체의 기능을 확장하여 사용자 경험을 개선하는 작업도 진행 중입니다.

#### 1.1. OpenEMR API 접근 전략 및 인증 심층 분석

* **OpenEMR API 인증**: 외부 애플리케이션의 API 접근에는 **OIDC(OpenID Connect) 규격을 따르는 OAuth 2.0 인증 방식**이 사용됩니다.
* **Django CDSS의 인증 절차**:
    1.  **클라이언트 등록**: `API_README.md`의 "Registration" 섹션에 따라 Django CDSS를 OpenEMR에 OAuth2 클라이언트로 먼저 등록하여 `client_id`와 `client_secret`을 발급받습니다.
    2.  **접근 토큰 획득**: 권장되는 "Authorization Code Grant" 방식을 사용하여, 사용자가 CDSS를 통해 OpenEMR에 로그인하고 권한 부여에 동의하면, CDSS는 발급받은 권한 코드로 OpenEMR 토큰 엔드포인트에 접근 토큰을 요청합니다.
    3.  **API 호출 시 토큰 사용**: 획득한 접근 토큰은 OpenEMR API 호출 시 HTTP `Authorization` 헤더에 `Bearer <토큰>` 형태로 포함됩니다.
* **OpenEMR 내부 코드 확인**: OpenEMR의 `oauth2/authorize.php` 파일은 OAuth2/OIDC 인증 서버의 핵심 로직을 처리하며, `apis/dispatch.php` 파일은 API 요청의 중앙 관문 역할을 하며 외부 요청에 대해 OAuth2 접근 토큰을 검증하고 권한을 확인합니다.
* **내부 API 호출과의 구분**: `tests/api/InternalApiTest.php` 파일은 OpenEMR 시스템 내부에서 CSRF 토큰을 사용하여 API를 호출하는 예시를 보여주나, 이는 외부 애플리케이션인 Django CDSS에는 해당되지 않습니다.
* **FHIR R4 표준 적극 활용**: OpenEMR이 지원하는 FHIR R4 표준을 통해 CDSS와 OpenEMR 간의 데이터(환자, 관찰, 진단, 처방 등)를 교환하는 것을 목표로 합니다. `API_README.md`에서도 FHIR API의 중요성을 언급하며 `FHIR_README.md`를 참조하도록 안내하고 있습니다. (*참고: FHIR_README.md 파일은 제공해주신 루트 디렉토리 목록에 포함되어 있습니다.*)
* **Django 연동 모듈 개발**: Django 백엔드에 `openemr_integration_module`을 개발하여 OpenEMR API 호출, OAuth2 인증 처리, 데이터 변환 등을 담당하도록 합니다.

#### 1.2. OpenEMR 환자 대시보드 개선: DICOM 뷰어 직접 임베딩

단순히 외부에서 데이터를 가져오는 것을 넘어, OpenEMR 사용자 경험을 직접 개선하기 위해 CT 등의 DICOM 영상을 OpenEMR의 환자 대시보드(예: "Medical Record Dashboard") 내에 바로 표시하는 기능을 통합하고자 합니다.

* **기존 DICOM 뷰어 활용**: OpenEMR은 `templates/dicom/dicom-viewer.html.twig` 파일을 통해 DWV.js 기반의 웹 DICOM 뷰어 기능을 이미 제공하고 있습니다.
* **직접 임베딩 전략**:
    * **`<iframe>` 활용**: 가장 현실적인 초기 접근법으로, 대시보드 내 특정 영역에 `<iframe>`을 두고, 이 iframe의 `src`를 동적으로 OpenEMR DICOM 뷰어 페이지 URL(영상 `doc_id` 및 CSRF 토큰 포함)로 설정하여 뷰어를 로드합니다. 동일 출처(OpenEMR 내부)이므로 외부 임베딩 시의 복잡한 인증 문제는 일부 완화될 수 있습니다.
    * *기타 고려 가능한 복잡한 방법*: AJAX로 뷰어 HTML 일부를 로드 후 DWV.js를 수동 초기화하거나, Knockout.js 컴포넌트로 뷰어를 재구성하는 방법도 있으나 초기 구현에는 많은 노력이 필요합니다.
* **OpenEMR 핵심 파일 수정**:
    * **중앙 제어 파일**: `interface/main/tabs/main.php` 파일은 환자 대시보드를 포함한 탭 인터페이스의 구성, JavaScript 뷰 모델 로드, Twig 템플릿 렌더링 등을 총괄하는 핵심 파일입니다.
    * **필요한 수정 범위**:
        * **PHP 로직**: `main.php` 또는 연관된 컨트롤러/서비스 파일에 현재 환자의 DICOM 영상 식별자(`doc_id`)를 조회하는 로직을 추가합니다.
        * **Twig 템플릿**: 대시보드 내용을 정의하는 Twig 템플릿 파일(주로 `interface/main/tabs/templates/` 내 위치)을 수정하여 `<iframe>`을 위한 공간을 마련하거나, 뷰어를 표시할 영역을 정의합니다.
        * **JavaScript (Knockout.js)**: `interface/main/tabs/js/` 내의 뷰 모델 파일들을 수정하여 `iframe`의 `src`를 동적으로 변경하거나, 새로운 탭/영역의 동작을 제어할 수 있습니다.
        * **메뉴 설정**: 만약 "영상" 탭을 새로 추가한다면 `interface/main/tabs/menu/menus/` 내의 JSON 설정 파일 수정이 필요할 수 있습니다.

### 2. 자체 CDW(Clinical Data Warehouse) 구축

* **CDW의 필요성**: AI 분석 및 연구를 위해 다양한 소스(EMR, LIS, PACS)의 임상 데이터를 통합, 정제, 분석 가능한 형태로 저장하는 CDW가 필수적입니다.
* **구축 결정**: 기존에 사용 가능한 CDW가 없어, 직접 구축하기로 결정했습니다.
* **ETL 파이프라인 (Django 기반)**:
    * **추출(Extract)**: `openemr_integration_module`을 통해 OpenEMR의 FHIR API 등에서 필요한 데이터를 추출합니다.
    * **변환(Transform)**: 추출된 데이터를 CDW 스키마에 맞게 정제하고, 필요시 익명화/가명화 처리합니다.
    * **적재(Load)**: 변환된 데이터를 CDW용 데이터베이스(예: PostgreSQL)에 저장합니다.
    * 이 모든 ETL 과정은 Django 내 `cdw_integration_module`에서 관리합니다.

### 3. LIS(검사실 정보 시스템) 연동: 오픈소스 SENAITE LIS 선택

* **LIS 데이터의 중요성**: 정확한 진단과 AI 모델 학습을 위해 LIS의 검사 결과 데이터는 매우 중요합니다.
* **OpenEMR의 LIS 연동 옵션 검토**: OpenEMR 자체의 LIS 연동 기능(Direct Lab Integrations, Laboratory Exchange Network - LEN, Custom Integrations)을 검토했습니다. LEN은 주로 결과 수신에 중점을 둔 기능임을 확인했습니다.
* **오픈소스 LIS 도입**: 자체적으로 관리 가능한 오픈소스 LIS를 도입하기로 결정했습니다.
* **SENAITE LIS 최종 선택**: 여러 오픈소스 LIS를 검토한 결과 (예: `jibrel/My-Awesome-healthcare` 목록의 OpenELIS, SENAITE), **SENAITE LIS**를 사용하기로 결정했습니다.
    * **SENAITE JSON API 활용**: SENAITE는 매우 잘 문서화된 JSON API를 제공하며, 이를 통해 외부 시스템과의 연동이 용이합니다. 이 API는 검사 의뢰(AnalysisRequest) 생성, 결과 조회 등 LIS의 핵심 기능을 프로그래밍 방식으로 제어할 수 있게 해줍니다.
    * **연동 전략**: Mirth Connect와 같은 통합 엔진을 사용하거나, Django 내 맞춤형 모듈을 개발하여 OpenEMR/CDSS와 SENAITE API 간의 데이터(오더, 결과) 흐름을 중계할 계획입니다.

### 4. PACS(의료영상정보시스템) 연동 (일반)

* CT, MRI 등 의료 영상 데이터는 암 진단 AI의 핵심 입력 데이터입니다.
* 오픈소스 DICOM 서버(Orthanc, DCM4CHEE 등)를 활용하고, Django에 PACS 인터페이스 모듈을 개발하여 DICOMweb 표준으로 영상 데이터를 연동할 계획입니다. (이는 대시보드 내 뷰어 연동과는 별개로 데이터 소스로서의 PACS 연동입니다.)

## OpenEMR 인터페이스(UI) 스타일 커스터마이징 환경

OpenEMR의 사용자 인터페이스를 일부 커스터마이징하거나 스타일을 수정해야 할 경우를 대비해 개발 환경 설정 방법을 파악했습니다.

* **UI 구성**: OpenEMR 인터페이스는 Bootstrap을 기반으로 SASS를 사용하여 구축되었으며, Gulp로 컴파일됩니다.
* **로컬 개발 환경**:
    1.  Node.js와 npm을 설치합니다.
    2.  OpenEMR 프로젝트의 `CONTRIBUTING.md` 파일에 설명된 대로 로컬 개발 환경을 설정합니다.
    3.  프로젝트 루트에서 `npm install`을 실행하여 필요한 패키지를 설치합니다.
    4.  `npm run dev` 명령어를 사용하면 `.scss` 파일 수정 시 자동으로 CSS 파일이 컴파일됩니다. `npm run dev-sync` (실험적)는 BrowserSync를 통해 실시간 미리보기를 지원합니다.

## AI 모듈 개발 및 통합 (Phase 4)

시스템 연동의 궁극적인 목표 중 하나는 AI 기반의 진단 지원 기능을 구현하는 것입니다.

* **AI 플랫폼 오케스트레이터**: Django 내에 AI 모델 호출, 데이터 전처리, 결과 관리를 담당하는 오케스트레이터를 구축합니다.
* **데이터 소스**: AI 모델은 OpenEMR(임상 정보), PACS(의료 영상), CDW(정제된 데이터, 오믹스 데이터 등), 그리고 SENAITE LIS(검사 결과)로부터 제공되는 데이터를 활용합니다.
* **Django의 역할**: 각 시스템 연동 모듈을 통해 AI 분석에 필요한 데이터를 수집하고, 이를 AI 모델에 전달합니다.
* **AI 결과 저장 및 활용**: AI 분석 결과는 FHIR DiagnosticReport 등의 표준 형식으로 OpenEMR에 저장되어 진료에 활용될 수 있도록 합니다 .

## 주요 기술 및 도구

* **EHR**: OpenEMR (FHIR API 및 OAuth 2.0 인증 활용, 내부 UI 커스터마이징)
* **백엔드 및 통합 허브**: Django (Python)
* **LIS**: SENAITE (JSON API 활용)
* **데이터베이스**: PostgreSQL (주 DB, CDW용 DB)
* **통합 엔진 (고려)**: Mirth Connect
* **데이터 표준**: FHIR, DICOMweb
* **UI 개발/커스터마이징**: SASS, Node.js, npm, Twig, Knockout.js
* **개발 환경**: Google Cloud VM, PuTTY

## 현재 상태 및 다음 단계

OpenEMR API의 OAuth2 인증 방식에 대한 이해를 바탕으로 Django CDSS에서의 클라이언트 로직 구현, SENAITE LIS와의 API 연동, 그리고 OpenEMR 자체 환자 대시보드 내에 DICOM 뷰어를 직접 임베딩하는 작업에 집중하고 있습니다. 특히 OpenEMR의 PHP 코드와 Twig 템플릿, Knockout.js 뷰 모델을 분석하며 대시보드 수정을 진행할 계획입니다.

## 마치며

오픈소스 소프트웨어와 표준화된 프로토콜을 적극적으로 활용하여 복잡한 의료 정보 시스템을 구축하는 것은 많은 도전과 학습을 필요로 합니다. 특히 기존 시스템의 내부 구조를 파악하고 기능을 확장하는 과정은 심도 있는 분석을 요구합니다. 하지만 이를 통해 더욱 유연하고 강력하며 투명한 시스템을 만들 수 있다고 믿습니다. 앞으로도 진행 상황을 꾸준히 공유하겠습니다. 읽어주셔서 감사합니다!