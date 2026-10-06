# JevPilot 출처 검토 및 적용 결정

작성일: 2026-10-06. 이 문서는 원안, 현재 코드, 참고 구현, 이번 보완 제안을 구분한다. 저장소 소스는 읽었지만 실제 모델 호출이나 브라우저 벤치마크는 수행하지 않았다.

## 1. 출처 목록

| ID | 자료 | 확인 범위 / 용도 |
| --- | --- | --- |
| S1 | `JevPilot_개인프로젝트_제안서_및_발표자료_Brief.docx` | 원본 본문·표·연결 링크 확인. §1~§6의 문제·역할·기능·3주 계획·평가 목표를 기준으로 유지 |
| S2 | `JevPilot 프로젝트 제안_발표자료.pdf` | 13페이지 원본. p.8~p.13의 문제 정의·역할·흐름·기술·일정·평가지표 확인; 도판과 숫자는 원본의 맥락을 유지 |
| S3 | [@0x_rody의 X 글](https://x.com/0x_rody/status/2104206841697779756) | X 직접 조회 실패. 검색에 노출된 [TwStalker 미러](https://ww1.twstalker.com/0x_rody)에서 해당 내용 확인. 정량 성능의 근거로 채택하지 않음 |
| S4 | [@zodchiii의 X 글](https://x.com/zodchiii/status/2101243146596384854), [작성자의 연결 글](https://zodchiii.substack.com/p/the-jev-setup-guide-how-to-get-maximum) | X 직접 조회 실패. 작성자 Substack의 검색 노출 본문으로 `The Jev Setup Guide: How to Get Maximum Quality for Minimum Cost` 확인. 공개 페이지 직접 open도 실패했으므로 접근 방식 기록 |
| S5 | [browser-use/jev-ultrafast](https://github.com/browser-use/jev-ultrafast/tree/1231850a0bf1a0c0341fe408ef1668dbbfdfac46) | README, agent.py, model.py, browser.py, snapshot.js의 앞 100줄, performance.md, LICENSE를 GitHub 연결로 확인 |
| S6 | [RatelXD/JevPilot](https://github.com/RatelXD/JevPilot/tree/0ad469bc222c65a02b616300ff0664dfa40e8476) | CONTRIBUTING·트리·기존 architecture·Accepted ADR·observer 확인, 앞선 동일 코드 검토의 provider/runtime와 대조. 현재 기능과 목표 기능을 분리 |

S3/S4의 ‘글 내용을 확인했다’는 표현은 그 글의 주장이 독립적으로 검증되었다는 뜻이 아니다. 이전 ‘링크 확인’ 대화의 요약은 출처 탐색에만 사용했고, 이번에 확보한 페이지·코드와 구분했다.

TypeSafe 공식 문서의 `patterns/fan-out`, `confidence`, `model-jaggedness/jev-1.13`도 재조회했지만 이번 접근에서는 본문을 확보하지 못했다. 따라서 이전 답변의 API 한도·가격·최신 버전을 그대로 확정하지 않았다. 구현 직전 공식 API/SDK 계약을 다시 확인해야 한다.

## 2. 원안에서 유지한 것

제안서의 중심은 ‘Jev를 사용했다’가 아니라 ‘반복적인 UI 선택을 별도 결정 모델에 위임했을 때 성공률·시간·호출 수·비용이 어떻게 바뀌는가’다. 고수준 계획·예외 복구는 LLM, 후보 선택은 Jev, 관측·실행은 Playwright라는 역할을 유지한다. [S1 §1~§4, S2 p.8~p.11]

원안에는 1주차 5~10개 과제 정의와 baseline·Mock 준비, 2주차 실제 Jev 연동과 최소 3개 과제 완주, 3주차 반복 평가·데모가 있다. 이번 문서는 이를 단계 완료 기준으로 세분화한 것이며 새로운 3주 일정을 약속하지 않는다. Docker와 간단한 대시보드는 원안의 예정 기술이지만 발표 직전 우선순위를 M1으로 낮춘 것은 이번 제안이다. [S1 §4~§6]

## 3. 두 X 글을 어떻게 반영했는가

### S3: 작은 판단을 먼저 처리하고 어려운 사례만 큰 모델로 보냄

미러에는 Jev가 먼저 분류·필터링하고 어려운 사례만 Claude로 넘긴다는 설명이 있다. 이 역할 분리는 JevPilot의 선택/재계획 분리를 설명하는 관련 사례로만 사용한다. ‘스탠퍼드 연구팀이 15분마다 1,000억 개 이상의 데이터 포인트를 처리한다’는 내용은 게시글의 주장까지 확인되었고, 해당 연구·데이터·측정 방법은 확보하지 못했다. 성능 목표나 입증 자료에서 제외한다.

### S4: 판단 호출의 구조와 검증 경계

작성자 글에서 참고한 내용은 같은 상태에 대한 독립 판단을 묶고, 새 관측이 필요한 판단은 분리하며, 질문·모델 버전·임계값을 관리한다는 원칙이다. confidence를 실행 권한이나 실제 완료 증거로 해석하지 않고 계산·날짜 처리·문자열 생성의 역할도 분리한다.

글의 예시 임계값과 ‘12.2배 저렴함’ 같은 수치를 JevPilot의 확정 설정이나 성능 목표로 채택하지 않는다. 묶음 질문·동적 후보·독립 검증의 적용 방법은 S5 실제 코드와 대조해 명세화했다.

## 4. `jev-ultrafast`에서 확인한 구현

검토 당시 main: `1231850a0bf1a0c0341fe408ef1668dbbfdfac46`. 아래 blob SHA는 실제 읽은 파일의 응답에서 얻었다.

| 파일 / blob SHA | 확인 내용 | JevPilot 적용 판단 |
| --- | --- | --- |
| `README.md` / `fa7d079f9192f8ad47d4016c0d534cfe72207f33` | 동적 action space, Jev operation/target 선택, 필요한 경우만 LLM 텍스트 생성 | 역할 분리 패턴 참고, 제품 전체 복제하지 않음 |
| `jev_ultrafast/agent.py` / `726d880bd95ca946106220a44c05a488a4f1da82` | predict/act 분리, 결정 1회 소비, stale 재관측, 실제 실행 뒤 기록, text context 일치 시 재사용 | 다단계 runtime와 중복 실행 방지에 참고 |
| `jev_ultrafast/model.py` / `d31174749e1451dbbd2a05f255f6f87fe296b122` | operation + operation별 target 질문, 선택한 target head만 검증·매핑, text JSON 검사 | Jev provider와 text provider 설계에 참고 |
| `jev_ultrafast/browser.py` / `38e3aced572de1da2959c86b6240a6e4b13f05c4` | Browser Harness/CDP 사용, 실제 DOM 대상·가림·freshness 검사, uncertain native select 별도 오류 | Playwright에서 동등한 가드를 설계하되 실행기는 교체하지 않음 |
| `jev_ultrafast/snapshot.js` / `cf748375d1b824e0444429018003dc2829dfa408` | 노드 identity, 의미 상태·주변 문맥, 가시 컨트롤과 상태 추출, 후보 제한 | Observer를 동적·provider 독립적으로 확장하는 참고 |
| `docs/performance.md` / `43f31cc5832054d59b0573e79b1be5a3023fdc31` | 비교 조건·측정 경계·실패한 개발 시도·제외 범위 설명 | 결과 보고의 투명성 참고, 수치를 목표로 복사하지 않음 |
| `LICENSE` / `271d8e2807d0d79aee85a231f5c9dbb007902783` | MIT, Copyright (c) 2026 Browser Use | 실제 코드 재사용 시 고지·출처 유지 |

### 잘못 가져오면 안 되는 차이

1. `Agent.__init__`의 `plan`은 입력 task를 담는 목록이다. 별도 LLM 고수준 planner가 아니다. JevPilot의 계획·재계획은 별도로 설계해야 한다.
2. `Agent`가 confidence를 기록한다고 해서 낮은 confidence에서 LLM planner로 전환하는 기능이 구현되어 있는 것은 아니다. JevPilot의 fallback은 원안에서 온 추가 요구다.
3. Agent의 DONE 상태와 독립적인 task verifier는 다르다. README/performance는 예제의 독립 검증을 설명한다. 일반 Agent의 DONE을 그대로 성공률로 세면 안 된다.
4. 제약 없는 기존 Chrome profile 사용을 JevPilot의 기본값으로 가져오지 않는다. 별도 테스트 context를 사용하는 것은 이번 보완 제안이다.
5. 원본은 action label을 생성 모델이 임의 selector로 변환해 실행하는 구조가 아니다. 관측된 실제 대상의 매핑을 코드가 관리한다.

### 성능 수치의 측정 경계

performance.md의 7.073초는 첫 prediction부터 accepted DONE까지다. 초기 브라우저 준비·초기 탐색과 독립적인 사후 검증은 측정 밖이다. 동일 조건의 runtime 비교는 한 과제를 세 쌍 반복한 결과이며, baseline LLM-only와 Jev-hybrid의 비교가 아니다.

따라서 ‘Jev가 LLM보다 25% 빠름’이나 ‘JevPilot은 7초에 완료해야 함’으로 해석하지 않는다. JevPilot은 마지막 독립 검증까지 포함하는 별도 task_e2e 지표를 사용하도록 제안했다. 참고 소스의 보조 text 모델 비용도 전체 작업 비용과 구별했다.

## 5. 현재 JevPilot과의 간극

현재 architecture 문서는 스스로 Initial Baseline으로 표시한다. Observer는 전달받은 selector 목록에서 CLICK 후보를 만드는 최소 구조다. 실제 provider 연동·다단계 상태·독립 task evaluator·완전한 비용 집계는 명세서에서 구현 목표로 분리했다. [S6]

기존 ADR은 공통 상태·후보를 통한 공정성을 요구한다. 온라인에서 서로 다른 행동이 나오면 이후 상태도 달라질 수 있다. 이번 문서는 온라인 end-to-end 조건 통제와 M1 동일 snapshot replay를 구분해 이 한계를 명시했다. 기존 ADR의 Accepted 상태나 원문은 수정하지 않았다.

## 6. 원본 자료 보존과 주의점

원본 파일은 변환·재작성하지 않고 SHA-256으로 식별한다. 배치 예정 경로는 다음과 같고, 실제 저장소 업로드 상태는 [원본 매니페스트](originals-manifest.json)에 기록한다.

| 원본 | docs 경로 | 바이트 |
| --- | --- | ---: |
| 제안서 + Gemini 발표 Brief | `docs/proposal/jevpilot-proposal-and-brief.docx` | 45668 |
| 제안 발표자료 | `docs/presentation/jevpilot-proposal-slides.pdf` | 1070101 |

SHA-256:

```text
fcb0fba97803290d12550556ed83e4a8f8765e505af704ed385d8b6d1cc21479  jevpilot-proposal-and-brief.docx
d3aa231cef76d209fc688ed73f1c720eba19ddcb922b90ea9388be9d77f0400c  jevpilot-proposal-slides.pdf
```

원본 검토에서 다음 차이를 확인했지만 파일을 수정하지 않았다.

- PDF p.5에는 콘텐츠가 아닌 빨간 X 형태의 도판이 보인다. 그 안에 원래 어떤 그림이 있어야 하는지는 추정하지 않았다.
- PDF p.3은 단일 호출 예시와 ‘실사용 수치’ 표기, p.4는 공급사 공개 수치 표기다. 단일 예시를 JevPilot benchmark 결과로 사용하지 않는다.
- DOCX Brief는 약 7분·권장 10장이라고 설명하지만 실제 PDF는 13페이지다. 실제 PDF 페이지 수를 10장으로 바꾸어 기록하지 않았다.
- 제안서의 ‘Jev가 계획을 무시할 수 없다’는 표현은 의도를 나타내지만, 제한된 후보만으로 의미적 계획 준수까지 보장되지는 않는다. 새 명세서에서는 코드 제약·결과 검증·잔여 위험을 별도로 설명했다. 원문은 보존했다.
- Early Access·예시 모델·가격·속도는 원본 작성 당시의 설명이다. 이번 업로드가 최신 사실 확인을 의미하지 않는다.

## 7. 결정 요약

**유지:** LLM planner + Jev selector + Playwright, 브라우저 우선, 공통 Action Space, confidence/실패 시 LLM 복구, 네 가지 핵심 평가지표.

**참고하여 보완:** 동적 DOM 후보, 한 관측의 operation/target 묶음 결정, 필요한 문자열만 생성, freshness·가림·중복 실행 방지, 독립 verifier, 세부 trace와 측정 경계.

**이번에 확정하지 않음:** 실제 API 버전·한도·가격, provider별 임계값, LLM 모델, 반복 수·성공률 차이의 허용 기준.

**제외:** S3의 미검증 대규모 처리량, 참고 구현의 속도를 JevPilot 목표로 사용, Browser Harness 교체, Kev/Lev/Laya·자체 학습·컨텍스트 압축 신규 기능.
