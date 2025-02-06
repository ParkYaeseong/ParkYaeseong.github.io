---
title: DML, DDL, DCL, TCL 이란?
date: 2025-01-29 7:59:00 +09:00
categories: [정보처리기사, 필기]
tags: [정보처리기사, 필기, 이론, DML, DDL, DCL, TCL]
---

| 명령어 종류 | 명령어 | 설명 |
|------------|--------|--------------------------------------------------|
| **데이터 조작어**  <br> (DML : Data Manipulation Language) | SELECT | 데이터베이스에 들어 있는 데이터를 조회하거나  <br> 검색하기 위한 명령어를 말하는 것으로 RETRIEVE 라고도 함 |
| | INSERT,  <br> UPDATE,  <br> DELETE | 데이터베이스의 테이블에 들어 있는 데이터에 변화를 가하는 종류 <br> (데이터 삽입, 수정, 삭제)의 명령어들을 말함 |
| **데이터 정의어**  <br> (DDL : Data Definition Language) | CREATE, <br>  ALTER,  <br> DROP,  <br> RENAME,  <br> TRUNCATE | 테이블과 같은 데이터 구조를 정의하는데 사용되는 명령어로 <br> (생성, 변경, 삭제, 이름변경)  <br> 데이터 구조와 관련된 명령어들을 말함 |
| **데이터 제어어**  <br> (DCL : Data Control Language) | GRANT, <br>  REVOKE | 데이터베이스에 접근하고 객체들을 사용하도록  <br> 권한을 주고 회수하는 명령어들을 말함 |
| **트랜잭션 제어어** <br> (TCL : Transaction Control Language) | COMMIT,  <br> ROLLBACK,  <br> SAVEPOINT | 논리적인 작업의 단위를 묶어서 DML에 의해 조작된 결과를  <br> 작업단위(트랜잭션)별로 제어하는 명령어를 말함 |


### DDL :
- PRIMARY KEY : 기본키 정의 
- FOREIGN KEY : 외래키 정의
- UNIQUE : 지정 속성은 중복값 가질 수 없음 
- NO ACTION : 변화가 있어도 조취를 취하지 않음
- CASCADE : 참조 테이블 튜플 삭제 시 관련 튜플 모두 삭제 및 속성 변경 시 속성값 모두 변경
- RESTRICTED : 타 개체가 제거할 요소를 참조중이면 제거를 취소
- SET NULL : 참조 테이블 변화 시 기본 테이블 관련 속성값 Null로 변경
- SET DEFAULT : 참조 테이블 변화 시 기본테이블의 관련 튜플 속성값을 기본값으로 변경
- CONSTRAINT : 제약 조건 이름 지정 
- CHECK 속성값에 대한 제약 조건 정의

### DML :
- INSERT INTO ~ VALUES : 튜플 삽입 
- DELETE FROM~ WHERE : 튜플 삭제
- UPDATE ~ SET ~ WHERE : 튜플 내용 변경 
- SELECT~FROM~WHERE : 튜플 검색
- DISTINCT : 중복 튜플 발견 시 그 중 첫번째 하나만 검색 
-  DISTINCTROW : 중복 튜플 제거 및 하나만 검색 (튜플 전체를 대상으로 검색)
- PREDICATE : 검색할 튜플 수 제한 
- AS 속성명 정의
- ORDER BY : 특정 속성 기준으로 정렬 후 검색할 때
- ASC : 오름차순 
- DESC : 내림차순 / 생략 시 오름차순
- GROUP BY : 특정 속성 기준 그룹화하여 검색할 때 사용 having절과 같이 사용되어야함