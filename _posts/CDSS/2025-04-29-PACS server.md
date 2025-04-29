---
title: PACS 서버 구축 가이드드
date: 2025-04-29 10:27:00 +09:00
categories: [CDSS]
tags: [CDSS, Orthanc, PACS server]
---
# 연구/개발용 Orthanc PACS 서버 구축 가이드: GCP 리눅스 VM과 Docker 활용

## 들어가며

의료 영상 데이터는 진단, 연구, AI 모델 개발 등 다양한 분야에서 핵심적인 역할을 합니다. 이러한 의료 영상 데이터를 효율적으로 저장, 관리, 조회하기 위한 시스템이 바로 PACS(Picture Archiving and Communication System)입니다. 하지만 상용 PACS 솔루션은 비용 부담이 크거나 특정 환경에 종속적인 경우가 많습니다.

이 글에서는 오픈소스 PACS 솔루션인 **Orthanc**를 **Google Cloud Platform(GCP)의 리눅스 가상 머신(VM)** 환경에 **Docker**를 사용하여 구축하는 과정을 상세하게 안내합니다. 연구, 개발 또는 학습 목적으로 자신만의 PACS 서버를 구축하고자 하는 분들에게 실질적인 도움이 되기를 바랍니다.

**왜 Orthanc인가?**

* **오픈소스:** 라이선스 비용 없이 자유롭게 사용하고 수정할 수 있습니다.
* **경량성:** 비교적 적은 리소스로도 구동 가능하여 개인 연구나 소규모 팀에 적합합니다.
* **확장성:** 플러그인 아키텍처를 통해 다양한 기능을 유연하게 추가할 수 있습니다. (예: DICOMweb, 데이터베이스 연동, 클라우드 스토리지, HL7 등)
* **REST API:** 강력한 REST API를 제공하여 다른 시스템과의 연동 및 자동화가 용이합니다.

**왜 Docker인가?**

* **환경 일관성:** 개발, 테스트, 운영 환경 간의 차이로 인한 문제를 최소화합니다.
* **간편한 배포 및 관리:** 복잡한 설치 과정 없이 컨테이너 이미지를 통해 손쉽게 애플리케이션을 배포하고 관리할 수 있습니다.
* **의존성 관리:** 필요한 라이브러리와 환경을 컨테이너 안에 격리하여 관리하므로 호스트 시스템과의 충돌을 방지합니다.

**왜 GCP인가?**

* **유연성 및 확장성:** 필요에 따라 VM 사양을 조절하고 스토리지를 확장하기 용이합니다.
* **관리 용이성:** 웹 콘솔을 통해 VM, 네트워크, 방화벽 등을 편리하게 관리할 수 있습니다.
* **다양한 서비스:** Cloud Storage, AI Platform 등 다른 GCP 서비스와의 연동 가능성을 열어둡니다. (이 글에서는 Compute Engine VM 중심으로 설명합니다.)

이 가이드에서는 GCP VM 생성부터 Docker 설치, Orthanc 설정 파일 준비 및 전송, Docker Compose를 이용한 Orthanc 및 PostgreSQL 실행, 기본 기능 확인, 그리고 보안 강화를 위한 비밀번호 해싱까지 전 과정을 단계별로 다룹니다.

**대상 독자:**

* 리눅스(Ubuntu/Debian 계열) 및 기본적인 터미널 명령어 사용에 익숙하신 분
* Docker 및 Docker Compose의 기본 개념을 이해하고 계신 분
* 연구, 개발, 학습 목적으로 개인 PACS 서버 구축에 관심 있는 개발자, 연구원, 학생

---

## 사전 준비

본격적인 구축에 앞서 다음 사항들이 준비되어 있어야 합니다.

1.  **Google Cloud Platform (GCP) 계정:** VM 인스턴스를 생성하고 관리할 수 있는 GCP 계정이 필요합니다.
2.  **SSH 클라이언트:** GCP VM에 원격으로 접속하기 위한 SSH 클라이언트가 필요합니다.
    * Windows: PuTTY, MobaXterm, Windows Terminal 등
    * macOS/Linux: 기본 터미널 사용 가능
3.  **SSH 키 페어:** GCP VM에 안전하게 접속하기 위한 SSH 공개키/개인키 페어. GCP에서 VM 생성 시 자동으로 생성하거나 기존 키를 등록할 수 있습니다. PuTTY를 사용한다면 `.ppk` 형식의 개인키 파일이 필요합니다.
4.  **(선택) 파일 전송 클라이언트:** 로컬 PC에서 작성한 설정 파일을 VM으로 전송하기 위한 도구가 필요합니다.
    * Windows: `pscp.exe` (PuTTY 설치 시 포함), WinSCP 등
    * macOS/Linux: `scp` 명령어 사용 가능
5.  **기본적인 리눅스 명령어 지식:** 파일/디렉토리 탐색, 텍스트 편집기(nano, vim 등) 사용, 권한 관리 등 기본적인 리눅스 명령어 사용법을 알고 있어야 합니다.

---

## 1단계: GCP Compute Engine VM 생성 및 설정

먼저 Orthanc 서버를 구동할 GCP VM 인스턴스를 생성합니다.

1.  **GCP Console 접속:** [Google Cloud Console](https://console.cloud.google.com/)에 로그인합니다.
2.  **Compute Engine > VM 인스턴스 > 인스턴스 만들기**로 이동합니다.
3.  **VM 설정:**
    * **이름:** VM을 식별할 수 있는 이름 (예: `orthanc-pacs-vm`)
    * **리전 및 영역:** 지리적으로 가깝거나 요구사항에 맞는 리전/영역 선택 (예: `asia-northeast3-a` - 서울)
    * **머신 구성:**
        * **시리즈:** E2 또는 N2D 등 범용 시리즈 선택
        * **머신 유형:** 초기에는 비용 효율적인 소형 타입으로 시작하는 것이 좋습니다 (예: `e2-small` 또는 `e2-medium`). 사용량에 따라 추후 변경 가능합니다. (최소 1 vCPU, 2GB RAM 이상 권장)
    * **부팅 디스크:**
        * **운영체제:** **Ubuntu** 선택
        * **버전:** **Ubuntu 22.04 LTS** (x86/64) 권장 (이 가이드 기준)
        * **부팅 디스크 유형:** 표준 영구 디스크 또는 SSD 영구 디스크 선택 (SSD 권장)
        * **크기:** 최소 20GB 이상 권장 (OS + Docker + Orthanc 메타데이터 고려). DICOM 파일은 별도 볼륨이나 클라우드 스토리지 사용 고려.
    * **ID 및 API 액세스:** 기본값 유지
    * **방화벽:** **HTTP 트래픽 허용** 및 **HTTPS 트래픽 허용** 체크 (추후 HTTPS 설정 시 필요). 추가로 Orthanc 통신을 위한 방화벽 규칙 설정이 필요합니다 (아래 참조).
4.  **네트워킹 > 네트워크 인터페이스:**
    * **외부 IP:** '임시' 또는 '고정 IP 주소 만들기' 선택 (고정 IP 권장)
5.  **SSH 키 설정:** (선택사항) GCP가 자동으로 관리하는 키 대신 직접 생성한 SSH 키를 사용하려면 '보안 > SSH 키 관리' 섹션에서 공개키를 등록합니다.
6.  **만들기** 버튼 클릭. VM 인스턴스 생성에는 몇 분 정도 소요될 수 있습니다.
7.  **방화벽 규칙 추가:** Orthanc 통신을 위한 포트를 열어주어야 합니다.
    * GCP Console에서 **VPC 네트워크 > 방화벽**으로 이동합니다.
    * **방화벽 규칙 만들기** 클릭.
    * **규칙 1 (Orthanc Web/API):**
        * 이름: `allow-orthanc-http` (예시)
        * 대상 태그: VM 인스턴스 생성 시 설정한 네트워크 태그 (또는 '네트워크의 모든 인스턴스' - 보안상 주의)
        * 소스 IP 범위: 접근을 허용할 IP 범위 (예: 내 IP 주소 `x.x.x.x/32`, 또는 테스트용 `0.0.0.0/0` - 보안 취약)
        * 프로토콜 및 포트: '지정된 프로토콜 및 포트' 선택, `tcp` 체크 후 포트 번호 `8042` 입력.
    * **규칙 2 (Orthanc DICOM):**
        * 이름: `allow-orthanc-dicom` (예시)
        * 대상 태그, 소스 IP 범위: 위와 동일하게 설정.
        * 프로토콜 및 포트: `tcp` 체크 후 포트 번호 `11112` 입력 (또는 `orthanc.json`에 설정할 DICOM 포트).
    * **만들기** 클릭하여 규칙 생성.
8.  **VM 접속 정보 확인:** 생성된 VM 인스턴스 목록에서 외부 IP 주소를 확인합니다.
9.  **SSH 접속:** PuTTY나 터미널을 사용하여 확인된 외부 IP 주소와 SSH 키(또는 GCP 사용자 계정)를 이용해 VM에 접속합니다.

    ```bash
    # 터미널 예시 (개인키 파일 사용)
    ssh -i ~/.ssh/your_private_key rsa-key-20250417@<VM_External_IP>

    # PuTTY 사용 시 Host Name에 IP 입력, Connection > SSH > Auth에서 ppk 파일 지정
    ```

---

## 2단계: 리눅스 VM에 Docker 및 Docker Compose 설치

이제 VM에 접속하여 Docker 환경을 구축합니다. (Ubuntu 22.04 LTS 기준)

1.  **패키지 목록 업데이트:**
    ```bash
    sudo apt update
    ```
2.  **필수 패키지 설치:**
    ```bash
    sudo apt install -y apt-transport-https ca-certificates curl gnupg lsb-release
    ```
3.  **Docker 공식 GPG 키 추가:**
    ```bash
    sudo install -m 0755 -d /etc/apt/keyrings
    curl -fsSL [https://download.docker.com/linux/ubuntu/gpg](https://download.docker.com/linux/ubuntu/gpg) | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    sudo chmod a+r /etc/apt/keyrings/docker.gpg
    ```
4.  **Docker APT 저장소 설정:**
    ```bash
    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] [https://download.docker.com/linux/ubuntu](https://download.docker.com/linux/ubuntu) \
      $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    ```
5.  **Docker 엔진 설치:**
    ```bash
    sudo apt update
    sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    ```
    * `docker-compose-plugin`이 설치되면서 최신 `docker compose` (V2) 명령어를 사용할 수 있게 됩니다.
6.  **Docker 서비스 시작 및 활성화:**
    ```bash
    sudo systemctl start docker
    sudo systemctl enable docker # 시스템 부팅 시 자동 시작
    ```
7.  **현재 사용자를 `docker` 그룹에 추가 (매우 중요!):** `sudo` 없이 `docker` 명령어를 사용하기 위해 필요합니다.
    ```bash
    sudo usermod -aG docker $USER
    ```
    **주의:** 그룹 변경사항을 적용하려면 **반드시 로그아웃 후 다시 SSH로 접속**해야 합니다.
8.  **(재로그인 후) 설치 확인:**
    ```bash
    # sudo 없이 실행되어야 함
    docker --version
    docker run hello-world
    docker compose version
    ```
    각 명령어 실행 시 버전 정보나 "Hello from Docker!" 메시지가 정상적으로 출력되면 성공입니다.

---

## 3단계: Orthanc 및 PostgreSQL 설정 파일 준비

Orthanc와 PostgreSQL 컨테이너의 동작 방식을 정의하는 설정 파일을 로컬 PC에서 미리 작성합니다.

1.  **`orthanc.json` 파일 작성:** Orthanc 서버의 상세 설정을 정의합니다. 아래는 필수적인 설정 예시입니다. **비밀번호는 반드시 안전한 값으로 변경하세요.**

    ```json
    // orthanc.json 예시
    {
      "Name" : "GCP-Research-PACS",
      "DicomAet" : "ORTHANC_GCP",
      "DicomPort" : 11112,
      "HttpPort" : 8042,
      "RemoteAccessAllowed" : true,
      "AuthenticationEnabled" : true, // 보안을 위해 true 강력 권장
      "RegisteredUsers" : {
        "admin" : "ReplaceWithStrongAdminPassword!", // <-- 강력한 비밀번호로 변경!
        "researcher" : "ReplaceWithStrongResearcherPassword!" // <-- 강력한 비밀번호로 변경!
      },
      "PostgreSQL" : {
        "Enable" : true,
        "Host" : "postgres", // docker-compose.yml의 서비스 이름
        "Port" : 5432,
        "Database" : "orthanc_db",
        "Username" : "orthanc_user",
        "Password" : "ReplaceWithStrongPsqlPassword!", // <-- 아래 docker-compose.yml과 동일하게!
        "EnableIndex" : true,
        "EnableStorage" : true
      },
      "DicomWeb" : {
        "Enable" : true,
        "Root" : "/dicom-web/",
        "EnableCors" : true, // 웹 앱 연동 시 필요
        "StowEnabled" : true,
        "QidoEnabled" : true,
        "WadoEnabled" : true
      },
      "StorageDirectory" : "/var/lib/orthanc/db", // 컨테이너 내부 저장 경로
      "StorageCompression": true,
      "DeleteAllowed" : true // 연구용이므로 허용 (주의 필요)
    }
    ```

2.  **`docker-compose.yml` 파일 작성:** Orthanc와 PostgreSQL 서비스를 정의하고 실행하는 방법을 명시합니다. **PostgreSQL 비밀번호는 위 `orthanc.json`의 `PostgreSQL.Password`와 동일하게 설정해야 합니다.**

    ```yaml
    # docker-compose.yml 예시
    services:
      orthanc:
        image: jodogne/orthanc-plugins:latest # 플러그인 포함 공식 이미지 권장
        container_name: orthanc_pacs_server
        ports:
          - "8042:8042"   # Orthanc Web/API 포트
          - "11112:11112" # Orthanc DICOM 포트
        volumes:
          - ./orthanc.json:/etc/orthanc/orthanc.json:ro # 설정 파일 마운트 (읽기 전용)
          - orthanc_storage:/var/lib/orthanc/db        # DICOM 파일 저장 볼륨
        environment:
          TZ: Asia/Seoul # 시간대 설정
        depends_on:
          - postgres
        restart: unless-stopped # 컨테이너 비정상 종료 시 자동 재시작

      postgres:
        image: postgres:15 # PostgreSQL 버전 지정
        container_name: orthanc_db
        environment:
          POSTGRES_DB: orthanc_db           # orthanc.json과 일치
          POSTGRES_USER: orthanc_user       # orthanc.json과 일치
          POSTGRES_PASSWORD: ReplaceWithStrongPsqlPassword! # <-- orthanc.json과 동일하게!
          TZ: Asia/Seoul
        volumes:
          - postgres_data:/var/lib/postgresql/data # DB 데이터 저장 볼륨
        restart: unless-stopped

    volumes:
      orthanc_storage: # DICOM 파일 영구 저장을 위한 명명된 볼륨
      postgres_data:   # PostgreSQL 데이터 영구 저장을 위한 명명된 볼륨
    ```

---

## 4단계: 설정 파일 VM으로 전송

로컬 PC에서 작성한 `orthanc.json`과 `docker-compose.yml` 파일을 VM으로 전송합니다. 여기서는 Windows 환경에서 PuTTY의 `pscp.exe`를 사용하는 예시를 보여줍니다.

1.  **VM에 대상 폴더 생성:** PuTTY로 VM에 접속하여 설정 파일을 저장할 폴더를 만듭니다.
    ```bash
    mkdir ~/orthanc_config
    ```
2.  **`pscp`로 파일 전송:** 로컬 PC의 명령 프롬프트(CMD) 또는 PowerShell에서 다음 명령어를 실행합니다. (파일 경로, 사용자명, VM IP, 개인키 경로는 실제 환경에 맞게 수정)

    ```powershell
    # orthanc.json 전송 (개인키 -i 옵션 사용 예시)
    & "C:\Program Files\PuTTY\pscp.exe" -i "C:\path\to\your\private_key.ppk" "C:\path\to\local\orthanc.json" rsa-key-20250417@<VM_External_IP>:~/orthanc_config/

    # docker-compose.yml 전송
    & "C:\Program Files\PuTTY\pscp.exe" -i "C:\path\to\your\private_key.ppk" "C:\path\to\local\docker-compose.yml" rsa-key-20250417@<VM_External_IP>:~/orthanc_config/
    ```
    * **주의:** Pageant를 사용 중이고 키가 로드되어 있다면 `-i` 옵션은 생략 가능합니다. `FATAL ERROR: No supported authentication methods available` 오류 발생 시 `-i` 옵션을 사용하거나 Pageant에 키를 로드해야 합니다.

---

## 5단계: Docker Compose로 Orthanc 실행

설정 파일 전송이 완료되면, VM에서 Docker Compose를 사용하여 Orthanc와 PostgreSQL 컨테이너를 실행합니다.

1.  **PuTTY로 VM에 접속합니다.**
2.  설정 파일이 있는 폴더로 이동합니다.
    ```bash
    cd ~/orthanc_config
    ```
3.  **Docker Compose 실행:** 백그라운드에서 서비스를 시작합니다.
    ```bash
    docker compose up -d
    ```
    * 처음 실행 시 필요한 이미지를 다운로드하므로 시간이 다소 걸릴 수 있습니다.
4.  **컨테이너 상태 확인:**
    ```bash
    docker compose ps
    ```
    * `orthanc_pacs_server`와 `orthanc_db` 컨테이너의 상태(STATUS)가 `Up` 또는 `Running`인지 확인합니다.
5.  **로그 확인 (오류 발생 시):** 컨테이너가 정상적으로 시작되었는지 로그를 통해 확인합니다. (Ctrl+C로 종료)
    ```bash
    docker compose logs -f orthanc postgres
    ```
    * Orthanc 로그에서 PostgreSQL 연결 성공 및 포트 리스닝 메시지를 확인합니다.

---

## 6단계: Orthanc 접속 및 기본 기능 확인

서버가 성공적으로 실행되었다면, 웹 브라우저와 DICOM 도구를 사용하여 기본적인 기능을 확인합니다.

1.  **Orthanc Explorer 접속:** 로컬 PC의 웹 브라우저에서 `http://<VM_External_IP>:8042` 주소로 접속합니다.
2.  **로그인:** `orthanc.json`의 `RegisteredUsers`에 설정한 사용자 이름과 **안전하게 변경한 비밀번호**로 로그인합니다.
3.  **기본 CRUD 테스트:**
    * **Create (업로드):** UI의 'Upload' 버튼을 통해 샘플 DICOM 파일을 업로드합니다. 또는 `storescu` (DCMTK) 같은 도구를 사용하여 VM의 DICOM 포트(`11112`)로 파일을 전송합니다.
    * **Read (조회):** UI에서 업로드된 환자/스터디 목록을 확인하고, 영상 뷰어를 통해 이미지를 확인합니다. REST API (`curl`)나 DICOMweb API, `findscu` 등을 이용한 조회도 테스트합니다.
    * **Update (수정):** REST API (`curl -X PUT /studies/{id}/modify ...`)를 사용하여 스터디 메타데이터 등을 수정해 봅니다. (주의 필요)
    * **Delete (삭제):** REST API (`curl -X DELETE /studies/{id}`)를 사용하여 테스트 데이터를 삭제해 봅니다. (주의 필요)

---

## 7단계: 보안 강화 - 비밀번호 해싱 (필수 권장)

`orthanc.json`에 일반 텍스트 비밀번호를 그대로 두는 것은 매우 위험합니다. Orthanc 실행 후 REST API를 사용하여 비밀번호를 해시값으로 변경해야 합니다.

1.  **PuTTY로 VM에 접속합니다.**
2.  **`curl` 명령어로 비밀번호 변경:** (예: 'researcher' 비밀번호 변경)
    ```bash
    # 'admin'의 현재 비밀번호(orthanc.json 값)와 researcher의 새 비밀번호 입력
    curl -u admin:ReplaceWithStrongAdminPassword! -X PUT http://localhost:8042/users/researcher \
    -d '{"Password": "EvenStrongerPasswordForResearcher123!"}'
    ```
3.  **모든 사용자의 비밀번호 변경:** `admin` 계정을 포함한 모든 등록된 사용자의 비밀번호를 위와 같은 방식으로 변경합니다. (admin 비밀번호 변경 시에는 다른 관리자 계정으로 하거나, admin 자신을 수정)
4.  **변경 확인:** 웹 UI에 접속하여 **새롭게 설정한 비밀번호**로 로그인이 되는지 반드시 확인합니다.

이제 Orthanc는 내부적으로 해시된 비밀번호를 사용하므로 훨씬 안전해졌습니다. `orthanc.json` 파일을 직접 수정하여 해시값을 저장할 수도 있지만, API를 통해 변경하는 것만으로도 실제 운영상의 보안은 강화됩니다.

---

## 8단계: 추가 보안 고려 사항

연구용 서버라도 기본적인 보안 조치는 중요합니다.

1.  **HTTPS 설정 (매우 중요):** 현재 설정은 HTTP를 사용하므로 웹 UI 로그인 정보나 API 통신 내용이 암호화되지 않습니다. 실제 사용 시에는 반드시 HTTPS를 설정해야 합니다.
    * **방법:** Orthanc 컨테이너 앞에 **리버스 프록시(Reverse Proxy)** 서버(예: Nginx, Caddy)를 별도의 Docker 컨테이너로 설정하는 것이 일반적입니다.
    * **Let's Encrypt:** 리버스 프록시 서버에서 Let's Encrypt를 사용하여 무료 SSL/TLS 인증서를 발급받고 자동 갱신하도록 설정할 수 있습니다.
    * **구성:** 외부 HTTPS(443 포트) 요청 -> 리버스 프록시 (SSL/TLS 처리) -> 내부 HTTP(8042 포트) 요청 -> Orthanc 컨테이너
2.  **방화벽 규칙 강화:** GCP 방화벽 규칙에서 소스 IP 범위를 `0.0.0.0/0` 대신 실제로 접근해야 하는 IP 주소 대역(예: 연구실 IP, VPN IP)으로 제한합니다. 불필요한 포트는 모두 차단합니다.
3.  **최소 권한 원칙:** VM 내에서 불필요하게 `root` 권한을 사용하지 않습니다. Docker 명령어는 `docker` 그룹에 속한 일반 사용자로 실행합니다.
4.  **정기적인 업데이트:** VM의 운영체제(Ubuntu), Docker 엔진, Orthanc 이미지, PostgreSQL 이미지 등 모든 소프트웨어의 보안 업데이트를 정기적으로 확인하고 적용합니다.

---

## 9단계: 백업 및 복구 전략

데이터 유실에 대비하여 정기적인 백업 계획을 수립하고 테스트해야 합니다.

1.  **백업 대상:**
    * **Orthanc 메타데이터:** PostgreSQL 데이터베이스 (`postgres_data` Docker 볼륨)
    * **DICOM 파일:** Orthanc 스토리지 (`orthanc_storage` Docker 볼륨)
    * **설정 파일:** `orthanc.json`, `docker-compose.yml` (Git 등으로 버전 관리 권장)
2.  **백업 방법:**
    * **PostgreSQL:** `docker compose exec postgres pg_dumpall -U orthanc_user > backup.sql` 과 같은 명령어로 DB를 덤프하여 백업 파일을 생성하고, 이 파일을 VM 외부(예: GCP Cloud Storage)에 안전하게 보관합니다. 정기적으로 자동화하는 스크립트를 작성할 수 있습니다.
    * **DICOM 파일 (Docker 볼륨):** Docker 볼륨 데이터가 저장되는 호스트 경로(보통 `/var/lib/docker/volumes/orthanc_config_orthanc_storage/_data`)를 `rsync`나 다른 백업 도구를 사용하여 주기적으로 백업합니다. 또는 GCP의 디스크 스냅샷 기능을 활용할 수도 있습니다.
3.  **백업 주기 및 보관:** 데이터 중요도와 변경 빈도에 따라 적절한 백업 주기(매일, 매주 등)와 보관 기간을 설정합니다.
4.  **복구 테스트:** 정기적으로 백업 데이터로부터 시스템을 복구하는 절차를 테스트하여 실제 상황에 대비합니다.

---

## 10단계: 모니터링

서버 상태와 성능을 지속적으로 관찰하여 문제를 조기에 발견하고 대응합니다.

1.  **기본 모니터링:** `docker stats` 명령어로 실행 중인 컨테이너의 실시간 리소스 사용량(CPU, 메모리, 네트워크 I/O)을 확인합니다. `docker compose logs`로 로그를 주기적으로 확인합니다.
2.  **시스템 모니터링:** GCP Cloud Monitoring 서비스를 활용하여 VM의 CPU, 메모리, 디스크, 네트워크 사용량을 모니터링하고 알림을 설정할 수 있습니다.
3.  **(고급) 애플리케이션 모니터링:** Prometheus/Grafana 스택을 별도로 구축하여 Orthanc 및 PostgreSQL의 상세 메트릭을 시각화하고 모니터링할 수 있습니다. (관련 Exporter 설정 필요)
4.  **(고급) 로그 관리:** 대규모 시스템이나 장기 운영 시에는 ELK(Elasticsearch, Logstash, Kibana) 스택이나 GCP Cloud Logging 같은 중앙 집중식 로그 관리 시스템을 도입하여 로그를 효율적으로 수집, 검색, 분석하는 것을 고려할 수 있습니다.

---

## 마무리

지금까지 GCP 리눅스 VM 환경에서 Docker와 Docker Compose를 사용하여 Orthanc PACS 서버를 구축하고 기본 설정을 완료하는 과정을 살펴보았습니다. 이 가이드가 여러분의 연구, 개발, 학습 활동에 도움이 되었기를 바랍니다.

여기서 구축한 환경은 기본적인 기능을 제공하며, 실제 임상 환경이나 민감한 데이터를 다루는 프로덕션 환경으로 사용하기 위해서는 앞서 언급한 **보안 강화(특히 HTTPS), 백업, 모니터링, 고가용성** 등을 더욱 철저하게 준비하고 검증해야 합니다.

Orthanc는 다양한 플러그인을 통해 기능을 확장할 수 있으므로, 필요에 따라 공식 문서나 커뮤니티를 통해 추가적인 정보를 탐색해 보시기 바랍니다.

궁금한 점이나 개선할 부분이 있다면 언제든지 댓글이나 연락을 통해 알려주세요.