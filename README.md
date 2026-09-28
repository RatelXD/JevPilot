# JevPilot

JevPilot는 **Hybrid Computer Use Agent** 연구를 위한 Python 베이스라인 프로젝트입니다.

## JevPilot이란?

프론티어 LLM이 계획(planning)을 담당하고, 반복적인 브라우저 액션 선택을 별도 selector(예: Jev)로 위임할 수 있는지 검증하는 실험 프레임입니다.

## 연구 질문

동일한 작업 성공률을 유지하면서,

- 지연 시간(latency)
- 프론티어 모델 호출 수
- API 비용

를 줄일 수 있는가?

## 아키텍처

User Goal -> LLM Planner -> Browser Observer / Action Builder -> Decision Provider -> Playwright Executor -> Verifier

- confidence가 임계값 미만이거나 실행/검증이 실패하면 planner로 되돌아가 재계획합니다.
- LLM-only와 Jev-hybrid는 **동일한 브라우저 상태 + 동일한 Action Space**를 입력으로 받습니다.

## 현재 상태 (초기 수직 슬라이스)

- Playwright로 로컬 브라우저 실행
- 결정적 테스트 페이지에서 ActionCandidate 추출
- MockDecisionProvider로 액션 선택
- 선택 액션 실행 및 상태 검증
- 실행 trace/telemetry 기록
- 실제 LLM/Jev 연동은 skeleton(TODO + 환경변수 기반 설정)

## 개발 환경

- Python 3.12+
- Playwright
- Pydantic
- pytest

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
python -m playwright install chromium
```

## 테스트 실행

```bash
pytest -q
```

## 수직 슬라이스 실행 예시

```bash
python benchmarks/tasks/run_minimal_slice.py
```
