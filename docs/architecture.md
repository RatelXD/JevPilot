# JevPilot Architecture (Initial Baseline)

JevPilot는 브라우저 작업 자동화를 위한 연구용 Hybrid Computer Use Agent 베이스라인입니다.

## 실행 흐름

User Goal -> LLM Planner -> Browser Observer / Action Builder -> Decision Provider -> Playwright Executor -> Verifier

실행/검증 실패 또는 결정 confidence가 임계값 미만이면 LLM Planner로 제어를 되돌려 재계획(replan)합니다.

## 핵심 원칙

- **Action Space 공정성**: LLM-only와 Jev-hybrid는 동일한 브라우저 상태와 동일한 ActionCandidate 집합을 입력으로 받습니다.
- **Provider 분리**: PlannerProvider, DecisionProvider로 런타임 결합도를 낮추고 교체 가능하게 유지합니다.
- **최소 수직 슬라이스**: 로컬 Playwright 브라우저, 결정적 페이지, 단일 액션 실행, 검증, 구조화된 trace를 제공합니다.

## 현재 포함 범위

- PlannerProvider 추상화 및 Static/LLM skeleton 구현
- DecisionProvider 추상화 및 Mock/LLM/Jev skeleton 구현
- ActionCandidate 추출(observer)
- Playwright 기반 action executor
- 텍스트 기반 verifier
- 실험 지표(성공/지연/calls/비용/fallback) telemetry 모델
