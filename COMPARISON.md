# JevPilot 비교와 검증 범위

2026-10-08 공개 소스 조회 기준. 아래 속도는 직접 재측정한 값이 아니다.

| 항목 | JevPilot | browser-use/jev-ultrafast | hqman/jev-browser-skill |
| --- | --- | --- | --- |
| 역할 | 독립 CLI, LLM 계획·재계획 + Jev 동작 선택 | 작은 Jev 실행 루프와 웹 inspector | 코딩 에이전트가 호출하는 CLI·daemon·skill |
| 텍스트 입력 | LLM이 필요한 입력값 생성 | 별도 소형 LLM이 TYPE_TEXT 때 생성 | needs_text로 멈추고 호스트가 jb reply로 입력 제공 |
| 브라우저 | 실행마다 새 Playwright context | Browser Harness로 기존 Chrome 프로필 연결 | 격리 Playwright Chromium, daemon이 세션 관리 |
| LLM 인증 | 공식 Codex 구독 프로필 또는 API gateway | README 예시는 OpenRouter 텍스트 모델 API 키 | 호스트 에이전트가 텍스트 담당, 별도 텍스트 모델 없음 |
| 완료 판단 | CLI의 URL·텍스트 독립 검증 필수 | 기본 Agent는 DONE을 done으로 전환; 예제에서 별도 결과 검증 | REVIEW 기반 needs_review는 모델 정책이며 강제 차단 보장 아님 |
| 네트워크 제한 | 허용 origin 밖의 모든 context 요청 차단 | 본 비교에서 정책 전체 미검증 | 허용 origin은 탐색만 제한, 기타 요청은 허용 |
| 관측·실행 | DOM 후보·문서/타깃 최신성 검증·실행 trace | 인덱스 동작 공간·DOM 참조·최신성/가림 검증 | 페이지 관측·액션·녹화·호스트 개입 |

## 해석

JevPilot와 ultrafast는 “관측된 후보에서 Jev가 동작을 고르고 LLM이 텍스트를
만든다”는 핵심이 유사하다. JevPilot는 별도 planner와 재계획, 실행 예산,
독립 완료 평가를 추가한다. 이것이 더 빠르다는 증거는 없다. 모델 호출과
검증 단계가 늘어 지연이 늘 수도 있다.

ultrafast README는 Google Flights 한 작업 7,073ms와 3회 비교 중앙값
9.450초→7.092초를 보고한다. 같은 환경에서 JevPilot와 비교한 수치가
아니므로 우열 근거로 사용할 수 없다. 기본 Agent의 DONE과 예제의
독립 검증도 구분해야 한다.

jb는 호스트 코딩 에이전트와 협력하는 방식이다. needs_text/reply와
daemon 세션 제어는 현재 JevPilot의 단발 실행과 다르다.

## 현재 JevPilot의 중요한 제한

이전 fixture 성공은 자연어 직접 입력이 설정된 브라우저 작업에 전달된다는
증거다. 자동 사이트 탐색이나 기차 시간표 조회가 완성됐다는 증거는 아니다.
현재 shell은 시작 URL 및 완료 조건을 환경 설정에서 받으며, 코레일 문자열에
대한 시작 URL 매핑만 있다. 코레일 실제 조회·정보 추출·답변을 검증하지 않았다.
기본 네트워크 origin 제한은 외부 CDN이나 별도 도메인을 사용하는 실사이트를
막을 수 있다. 범용 검색·로그인 세션·추가 질의·작업 재개를 완성했다고 볼 수 없다.

이번 실제 확인에서도 `printf '/exit\n' | uv run jevpilot`는 exit 2와
사용법 오류를 반환했다. 현재 프롬프트는 `jevpilot shell`로 열어야 한다.
이전 답변의 “Codex처럼 명령 하나로 시작” 완료 주장은 정확하지 않았다.

## 출처

- https://github.com/browser-use/jev-ultrafast
- https://github.com/browser-use/jev-ultrafast/blob/main/jev_ultrafast/agent.py
- https://github.com/browser-use/jev-ultrafast/blob/main/docs/performance.md
- https://github.com/hqman/jev-browser-skill
- https://github.com/hqman/jev-browser-skill/blob/main/CHANGES.md

## 이번 설정 변경 검증

- 저장된 ChatGPT 구독 모델과 `JEVPILOT_JEV_API_KEY`만으로 provider
  구성을 만든다. Jev 모델 기본값은 `jev-latest`, 두 confidence 임계값은
  `0.5`다. 비용 한도와 confidence 임계값은 다른 개념이다.
- Jev 키 존재 여부로 shell의 hybrid 실행을 선택한다.
- `.env`와 `.env.example`에서 LLM/Jev 테스트 지출 한도 설정을 제거했다.
  단계·시간 등의 실행 제한과 benchmark 내부 지출 제어는 유지한다.
- 최종 `uv run pytest -q`: **155 passed in 21.66s**, exit 0.
- 변경 Python 파일 여섯 개 LSP 오류 0건.
- 사용자 재개 승인 후 실제 구독 LLM + Jev 실행 성공(exit 0).
  `artifacts/runs/773b24c0057d4990b2c1c093c96d8cdb.json`에서 두 provider와
  Aurora URL·화면 텍스트 독립 검증 통과를 확인했다. LLM 1회, Jev 1회 호출.
  Jev 실제 응답 모델은 `jev-1.13.0`, 표시된 추정 비용은 $0.000042714였다.
- 실행 시 LLM provider/key/model과 Jev model/threshold 환경변수를 해제하고
  키만 있는 `.env`를 로드했다. fixture 서버는 종료했고 브라우저는 실행기가 정리했다.
  이는 provider 설정과 로컬 작업 완주 증거이며 코레일 실사이트 성공 증거는 아니다.
