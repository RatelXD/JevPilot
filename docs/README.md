# JevPilot 문서

## 구현 준비

| 문서 | 내용 |
| --- | --- |
| [요구사항 명세서](requirements.md) | 원안·참고 구현·보완 제안을 구분한 FR/NFR, 수용 테스트, 비교 실험, 구현 우선순위 |
| [구현 설계 프롬프트](prompts/implementation-design.md) | 코딩 에이전트가 현재 저장소를 읽고 최소 변경 설계를 작성하도록 하는 프롬프트. 코드 구현은 별도 단계 |
| [출처 검토](references/source-review.md) | 두 X 글의 확인 경로·한계, jev-ultrafast 코드 분석, 원본 자료와 설계의 차이 |
| [기존 아키텍처](architecture.md) | Initial Baseline 설명. 신규 명세서의 목표 기능이 구현되었다는 뜻이 아님 |
| [공통 Action Space ADR](adr/0001-shared-action-space.md) | 기존 Accepted 결정; 원문 유지 |

## 제안서와 발표자료

원본의 내용과 파일 형식을 보존한다. 위치와 SHA-256, 실제 업로드 상태는 [원본 매니페스트](references/originals-manifest.json)를 확인한다.

| 자료 | 배치 경로 |
| --- | --- |
| 개인 프로젝트 제안서 + 발표자료 제작 Brief | `docs/proposal/jevpilot-proposal-and-brief.docx` |
| 제안 발표자료 PDF (13페이지) | `docs/presentation/jevpilot-proposal-slides.pdf` |

매니페스트가 `pending_binary_upload`이면 원본은 문서 패키지에만 포함되어 있고 저장소 업로드는 아직 완료되지 않은 상태다. 빈 파일이나 텍스트로 대체한 DOCX/PDF는 원본으로 취급하지 않는다.

## 문서 상태

2026-10-06 작성한 요구사항과 설계 프롬프트는 구현 전 초안이다. 외부 자료의 수치는 JevPilot의 실측 결과가 아니며, 테스트·모델 연동·성능 개선은 실제 실행 기록으로 별도 확인해야 한다. 개발 작업 시 [기여 가이드](../CONTRIBUTING.md)를 함께 읽는다.
