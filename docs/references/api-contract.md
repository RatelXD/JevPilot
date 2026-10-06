# 실제 API 계약 확인

확인일: 2026-10-06.
실제 계정 접근과 모델 호출은 아직 검증하지 않았다.

## Jev

공식 출처:

- https://docs.typesafe.ai/api.md
- https://docs.typesafe.ai/models.md
- https://docs.typesafe.ai/confidence.md
- https://docs.typesafe.ai/patterns/fan-out.md
- https://docs.typesafe.ai/sdk/python/api/retries.md

| 항목 | 문서에서 확인한 계약 |
|---|---|
| 요청 | `POST https://api.typesafe.ai/v1/systemone` |
| 인증 | Bearer API key |
| 본문 | `state`, `model`, `questions` |
| Choice 질문 | `type`, `instructions`, `criteria` |
| Choice 응답 | `type`, `choice`, `probabilities`, `confidence` |
| 응답 envelope | `model`, `answers`, `usage` |
| usage | `input_tokens`, `output_tokens` |
| Choice 상한 | 255개 |
| 문맥 | 총 64k tokens; state와 가장 긴 질문의 합은 32k |
| 고정 모델 | 문서상 `jev-1.13.0`; 실제 계정 접근은 별도 확인 |
| 오류 | 401 인증, 422 요청 검증, 429 rate limit, 529 과부하 |

Choice 최소 개수와 요청당 질문 수의 숫자 상한은 확인하지 못했다.
질문은 같은 state에서 독립적으로 실행된다.
한 질문이 다른 질문의 답을 읽는다고 가정하지 않는다.

2026-10-06에 확인한 공식 OpenAPI 스키마
(`https://api.typesafe.ai/openapi.json`)에는 `/v1/systemone`과 `/v1/models`
만 있으며, 계정 누적 usage나 billing endpoint는 없다. JevPilot은 각
응답의 input token 수에 공식 input 요금 `$0.042/M`을 적용하고 output은
무료로 계산해 CallEvent의 `estimated_cost_usd`에 기록한다. TypeSafe 응답은
청구액 필드를 제공하지 않으므로 `billed_cost_usd`는 unknown으로 유지한다.
사용자가 알려준 35,569 input tokens는 `$0.001493898`로 계산되어 보고된
`$0.0015`와 반올림까지 일치했다. benchmark의 `jev_spend_reserved_usd`는
실제 청구액이 아닌 별도의 보수적 예산 reservation이다.

confidence는 분포의 집중도다.
실행 권한이나 실제 작업 성공률로 해석하지 않는다.
operation과 target 값을 따로 기록한다.

모델 페이지의 입력 가격은 조회 당시 $0.042/M tokens다.
출력은 무료로 표시되어 있다.
가격 적용일은 페이지에서 확인하지 못했다.
usage 기반 계산은 추정값이며 청구액이 아니다.
응답에 실제 청구액 필드가 있다고 가정하지 않는다.

HTTP 문서는 429와 529의 지수 backoff를 안내한다.
SDK 기본 재시도는 별도이므로 숨은 전송이 지표에서 빠지지 않게 해야 한다.
실제 구현은 각 전송 시도와 비용 상태를 기록한다.

## LLM: Chosun University API Gateway

The user selected the Chosun University API Gateway so one API key can route
to multiple model providers. The key belongs in the local `.env` beside the Jev
key; never read it into logs, messages, or committed files.

Official sources:

- https://docs.mindlogic.ai/docs/chosun-ac/api-gateway/getting-started/overview
- https://docs.mindlogic.ai/docs/chosun-ac/api-gateway/getting-started/authentication
- https://docs.mindlogic.ai/docs/chosun-ac/api-gateway/getting-started/models
- https://docs.mindlogic.ai/docs/chosun-ac/api-gateway/reference/chat-completions
- https://docs.mindlogic.ai/docs/general/baze/product/model-credits
- https://docs.mindlogic.ai/docs/chosun-ac/api-gateway/reference/credits
- https://docs.mindlogic.ai/docs/chosun-ac/api-gateway/reference/usage
- https://docs.mindlogic.ai/docs/chosun-ac/api-gateway/reference/errors

The gateway documents a shared base URL,
`https://factchat-cloud.mindlogic.ai/v1/gateway`, Bearer-key authentication,
`GET /models/`, and OpenAI-compatible
`POST /chat/completions/`. Chat Completions supports strict JSON Schema output.
The user selected model ID `gpt-6.1-sol`; account access remains unverified.

The user chose Jev alias `jev-latest` for the initial test. TypeSafe currently
maps it to `jev-1.13.0`; the trace records the response model so alias movement
is visible. The user selected exploratory operation/target thresholds of
0.5/0.5. Pin `jev-1.13.0` for later comparisons that require a fixed version.

Gateway usage and the published per-model rates are expressed in credits.
The model-credit page states input/output credits per 1,000 tokens and is dated
2026-09-30. The fetched gateway docs do not specify a USD-to-credit conversion
or per-chat USD charge. The user approved up to 10,000 Gateway credits and a
separate $1 Jev reservation for benchmark runs. These are test-only controls;
the product CLI has no JevPilot-imposed cost cap. Report Gateway spending in
credits and leave USD unknown unless the provider supplies a verified amount.

The user also requires a selectable ChatGPT subscription path for the final
project demo. `codex_subscription` uses Codex's existing login and must never
receive the Gateway or Jev API key. The Codex subscription login is not
verified.

Live Gateway evidence:

- `artifacts/runs/f52d8c1a25fa459188bdfcd9541bc872.json`: T-01 `llm_only`
  independently succeeded with four Gateway HTTP attempts and 6.082 estimated
  credits.
- `artifacts/runs/65c82199a806425ca0df60f918ba22de.json`: T-02 `llm_only`
  independently succeeded with four Gateway HTTP attempts and 8.558 estimated
  credits.
- `artifacts/runs/fa1f3afad349494ea973bf709cc7abcc.json`: T-03 `llm_only`
  independently succeeded with ten Gateway HTTP attempts, one replan, and
  16.776 estimated credits.

Each artifact reports `gpt-6.1-sol` as both the requested and response model.
The Gateway balance was 9,888.05 credits before T-01 and 9,856.64 after T-03;
the three verified runs used a net 31.41 account credits (31.416 estimated).
USD billing remains unknown.

The earlier combined smoke/compare process was terminated after about five
minutes with `EPIPE` before writing its artifact. Local fixture logs show task
navigation, but there is no trace proving provider outcomes or Jev request
count. Its process-local Jev reservation ledger was lost, so further Jev calls
require checking TypeSafe account usage or renewed user authorization. A
sanitized environment check confirmed the required key variables are present
without displaying their values.

## 비밀정보와 실행 정책

키 값은 문서나 대화에 기록하지 않는다. Gateway key는 Jev key와 함께
local `.env`에 넣고, `.env`는 권한 `0600`과 git ignore를 유지한다.
CLI는 `uv --env-file .env`로 값을 process에 전달한다. Gateway 키는
`JEVPILOT_LLM_API_KEY`, TypeSafe 키는 `JEVPILOT_JEV_API_KEY`로 둔다.

제품 CLI에는 JevPilot 자체 비용 상한을 두지 않는다. 선택된 Gateway 계정
또는 ChatGPT 구독이 실제 사용량을 처리한다. 10,000 Gateway credits와 Jev
$1 reservation은 benchmark 전용이다. 모델 key가 없거나 오류가 나면 fake
응답으로 전환하지 않는다. 실제 비교의 Jev 단계는 앞선 중단 run의 사용량
확인이 끝날 때까지 대기한다.
