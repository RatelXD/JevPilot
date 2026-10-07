# JevPilot

JevPilot는 사용자의 목표를 받고 브라우저에서 작업을 수행하는 에이전트입니다.
LLM은 작업을 계획하고 입력 문구를 만듭니다. Jev는 다음 브라우저 동작을
선택할 수 있습니다.

제품 실행 모드는 두 가지입니다.

- `llm_only`: LLM이 각 동작을 선택합니다.
- `jev_hybrid`: LLM이 계획과 입력 문구를 만듭니다. Jev가 관측된 동작 후보 중 하나를 선택합니다.

## 연구 목표

목표는 작업 성공률을 낮추지 않으면서 LLM-only보다 시간과 비용을 줄이는
것입니다. 결과는 같은 작업을 여러 번 실행해 측정합니다. 한 번의 실행이나
소규모 실험만으로 일반적인 성능을 주장하지 않습니다.

## 브라우저 작업 실행

JevPilot에는 시작 URL, 작업 목표, 완료 조건이 필요합니다. 별도의 검증기가
완료 조건을 확인합니다. 모델이 `DONE`을 선택했다는 이유만으로 작업을
성공 처리하지 않습니다.

예시:

```bash
uv run --env-file .env jevpilot run \
  --mode jev_hybrid \
  --url http://127.0.0.1:8000/search \
  --goal "Search for Aurora and open its details page." \
  --verify-url-path /items/aurora \
  --verify-text "Aurora details" \
  --live
```

완료 조건은 다음과 같습니다.

- `--verify-url-path`: 마지막 URL 경로가 지정한 값과 정확히 같은지 확인합니다.
- `--verify-text`: 페이지 본문에 지정한 문구가 보이는지 확인합니다.

조건을 두 개 모두 지정하면 둘 다 통과해야 합니다. 실행 결과는
`artifacts/runs/` 아래 JSON 실행 기록(trace)으로 저장합니다. 기록에는 모델
호출, 브라우저 동작, 검증 결과, 사용량과 확인 가능한 비용이 들어갑니다.
원본 prompt와 생성된 입력 문구는 저장하지 않습니다.

각 실행은 새 브라우저 context를 사용합니다. 기본적으로 시작 URL의 origin만
허용합니다. 추가 origin은 `--allow-origin`으로 각각 지정합니다. 허용하지
않은 origin으로 가는 요청은 차단합니다. 작업 대상 사이트의 개인 로그인 흐름,
결제, 삭제, 외부 메시지 전송은 현재 지원 범위가 아닙니다.

## LLM과 Jev 설정

`JEVPILOT_LLM_PROVIDER`로 LLM 실행 경로를 선택합니다.

- `gateway`: Chosun University API Gateway를 통해 여러 모델을 사용합니다.
- `chatgpt-subscription`: 공식 Codex CLI 로그인으로 ChatGPT 구독 모델을 사용합니다.

Gateway를 사용할 때는 Gateway 키와 Jev 키를 로컬 `.env`에 둡니다. Gateway
키는 Codex 하위 프로세스에 전달하지 않습니다.

```text
JEVPILOT_LLM_PROVIDER=gateway
JEVPILOT_LLM_API_KEY=<Chosun Gateway API key>
JEVPILOT_LLM_MODEL=gpt-6.1-sol
JEVPILOT_JEV_API_KEY=<Jev API key>
JEVPILOT_JEV_MODEL=jev-latest
JEVPILOT_JEV_OPERATION_THRESHOLD=0.5
JEVPILOT_JEV_TARGET_THRESHOLD=0.5
```

현재 TypeSafe는 `jev-latest`를 `jev-1.13.0`으로 해석합니다. 응답의 실제 모델
ID를 trace에 기록합니다. 별칭은 이후 다른 모델을 가리킬 수 있습니다.

ChatGPT 구독 로그인과 모델 선택:

```bash
uv run jevpilot shell
/login
/model
```

`/login`은 공식 `codex login`을 사용하고 OAuth 주소를 터미널에 표시합니다.
자동으로 Chromium을 열지 않습니다. Windows 브라우저에서 주소를 열고 인증을 마친 뒤
터미널로 돌아옵니다. Codex는 인증 정보를 `~/.jevpilot/codex`에 저장합니다.
JevPilot는 `~/.codex`에서 인증 정보를 복사하거나 토큰을 읽지 않습니다.
`/model`은 로그인한 Codex 프로필의 표시 가능한 모델을 조회하고
`chatgpt-subscription/<model-id>` 형식으로 선택합니다. 선택은
`~/.jevpilot/settings.json`에 저장하고 다음 `jevpilot run`에서 사용합니다.
선택과 함께 계정이 보고한 추론 강도 및 Fast 지원 상태도 저장합니다.
`/model` 목록은 지원하는 추론 강도와 Fast 여부를 표시합니다.
프롬프트 입력창에서 `Shift+Tab`을 누르면 현재 입력 문자열과 커서를 보존한 채
추론 강도가 바뀝니다. 프롬프트 입력 중 `Shift+Tab`은 입력 문장을 보존하며 지원되는
`low → medium → high → xhigh → max` 순으로 순환합니다. `/fast`, `/fast on`,
`/fast off`로 현재 모델의 Fast service tier를 전환합니다. Fast는 계정
카탈로그가 해당 모델을 지원한다고 보고한 경우에만 켤 수 있습니다.
headless 실행에서 `gateway`를 명시하면 환경변수 모델 설정을 사용해 저장된 구독 선택을
무시합니다. `chatgpt-subscription`은 저장 모델을 우선하고, 선택이 없으면 환경 모델을
사용합니다. 모델 목록은 선택 후보이며 실제 모델 접근은 첫 inference 요청으로 확인됩니다.
저장된 모델이 현재 계정 카탈로그에서 사라지면 실행 전에 선택을 다시 요구합니다.

uv는 `.env`를 자동으로 읽지 않습니다. 명령에 `--env-file .env`를 지정하거나
환경변수를 직접 주입합니다. `.env`는 Git에서 제외하고 권한을 `600`으로
설정합니다. 기존 `.env`에는 Jev 키가 있을 수 있으므로 `.env.example` 내용을
합칠 때 기존 파일을 덮어쓰지 않습니다.

제품 CLI는 JevPilot 자체 비용 상한을 적용하지 않습니다. 실제 비용은 선택한
API 계정이나 구독에서 청구됩니다. Jev 응답마다 입력/출력 token 수와 공개된
입력 요금으로 계산한 USD 추정치를 출력하고 실행 기록에 저장합니다. TypeSafe
API는 실제 청구액을 반환하지 않으므로 `billed_cost_usd`는 알 수 없는 값으로
둡니다. Gateway 사용량은 credits로 표시되며 USD 환산값은 알 수 없습니다.

## 두 모드 비교

반복 실험에는 작업 실행기를 사용합니다.

```bash
uv run --env-file .env python benchmarks/tasks/run_suite.py \
  --mode compare \
  --task all \
  --repeat 5 \
  --smoke-gate \
  --live
```

실행기는 같은 작업을 두 모드에서 실행하고 반복마다 실행 순서를 바꿉니다.
실패와 제한 시간 초과를 포함해 모든 실행을 저장합니다.

| ID | 작업 |
| --- | --- |
| `T-00` | 목록에서 항목을 엽니다. |
| `T-01` | 검색어를 입력하고 결과 항목을 엽니다. |
| `T-02` | North 필터를 선택하고 결과 항목을 엽니다. |
| `T-03` | 두 단계 synthetic form을 채우고 제출합니다. |
| `T-04` | 검색어를 입력하고 자동완성 항목을 엽니다. |
| `T-05` | 두 필터를 선택하고 결과 항목을 엽니다. |

`--task all --repeat 5`는 5개 작업 × 2개 모드 × 5회로 비교 실행 50회를
예약합니다. `--smoke-gate`를 지정하면 먼저 T-01, T-02, T-03을 양 모드에서
실행합니다. 사전 점검 6회가 모두 독립 검증에 통과하고 Gateway와 Jev 테스트
예산에 여유가 있을 때만 50회 비교를 시작합니다. 두 단계는 같은 실행 및
예약 예산을 사용합니다.

Benchmark에만 테스트 지출 제한을 적용합니다. Gateway 계정 잔액은 10,000
credits 이하이어야 하며 Jev 요청은 총 $1까지 보수적으로 예약합니다. 제품
CLI에는 이 제한을 적용하지 않습니다. 실행기는 시작 전후 Gateway 잔액을
기록합니다. Jev 예약액은 실제 청구액이 아닙니다. 기본 예약액은 요청당
$0.01입니다.

테스트 fixture에는 가상 데이터만 사용합니다. CLI와 작업 실행기는 실제 model
provider를 호출합니다. 자동 테스트는 특정 동작과 오류를 검사할 때만 로컬
브라우저와 scripted response를 사용합니다. `mock` 제품 모드는 없습니다.

두 모드는 첫 동작 이후 서로 다른 페이지 상태를 만들 수 있습니다. 따라서
비교는 전체 시스템의 결과를 측정합니다. 모든 단계에서 두 모드가 같은 상태를
봤다고 주장하지 않습니다.

## 현재 지원 범위

JevPilot는 관측된 HTML/ARIA 항목에서 다음 동작을 지원합니다.

- `CLICK`
- `TYPE_TEXT`
- 관측한 native select option의 `SELECT`
- 주 viewport의 `SCROLL_UP`, `SCROLL_DOWN`
- 페이지 상태 변화에 대한 제한된 `WAIT`
- `DONE`
- `BLOCKED`

Observer는 현재 페이지의 후보를 만듭니다. Executor는 후보를 관측한 DOM
node에 연결합니다. 동작을 실행하기 전에 document와 target이 최신 상태인지,
target이 보이고 활성화됐는지, hit test를 통과하는지 확인합니다. 각 결정은
한 번만 실행합니다. 변경이 적용됐는지 알 수 없으면 페이지 상태를 확인하고
자동으로 mutation을 다시 실행하지 않습니다.

현재 범위 밖에는 file upload, 새 탭, 운영체제 제어, 개인 로그인
세션, 결제, 삭제, 외부 메시지 전송이 있습니다. `iframe`, canvas 등 지원하지
않는 페이지 요소는 임의로 조작하지 않고 안전하게 실패합니다.

## 개발 및 테스트

Python 3.12 이상, uv, Pydantic, Playwright, pytest를 사용합니다.

```bash
uv sync --extra dev
uv run python -m playwright install chromium
uv run pytest -q
```

테스트는 실제 model API를 호출하지 않습니다. 실제 실행에는 `--live`, 명시적인
환경 설정, 승인된 Jev 사용 예약이 필요합니다. 반복 비교의 실제 결과를
확인하기 전에는 성능이 개선됐다고 주장하지 않습니다.

상위 작업 폴더의 `JevPilot-implementation-design.md`와 `JevPilot-plan.md`에서
설계와 구현 계획을 확인할 수 있습니다.
