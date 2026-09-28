# ADR 0001: LLM-only와 Jev Selector의 공통 Action Space 사용

- 상태: Accepted
- 날짜: 2026-09-28

## 문맥

JevPilot의 핵심 연구 질문은 반복적인 브라우저 액션 선택을 Jev로 위임했을 때,
성공률을 유지하면서 지연/프론티어 모델 호출/API 비용을 줄일 수 있는지입니다.

## 결정

LLM-only baseline과 Jev-hybrid 모두에 대해 동일한 브라우저 상태 관측 결과와 동일한 ActionCandidate 집합(Action Space)을 제공한다.

## 근거

1. Selector 외 요인의 영향을 최소화해 비교 공정성을 확보한다.
2. 성능 차이를 planner/observer 변화가 아닌 decision provider 차이로 귀속할 수 있다.
3. 실험 재현성과 통계적 해석 가능성을 높인다.

## 결과

- Observer/Action Builder는 provider-agnostic하게 유지한다.
- DecisionProvider는 동일한 입력 계약(typed candidates + plan context)을 공유한다.
- 추후 Jev/LLM provider 교체 시 런타임 핵심 경로 수정이 최소화된다.
