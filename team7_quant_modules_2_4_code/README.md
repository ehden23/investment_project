# 팀7 자산운용 프로젝트 — 모듈 1~4

이 프로젝트는 다음 흐름을 재현하는 연구용 Python 코드입니다.

1. **모듈 1:** 시점별 KOSPI 보통주 투자 유니버스 구성
2. **모듈 2:** 가치·퀄리티·모멘텀 스코어 및 상위 30종목 선정
3. **모듈 3:** 동행지수 순환변동치를 이용한 4개 경기국면 판별
4. **모듈 4:** 직전 7년 상대성과를 이용한 국면별 가치·퀄리티 가중치

웹 화면, 자동 데이터 수집, 실제 주문은 포함하지 않습니다. DataGuide·KOSIS 자료를 내려받아
표준 열로 변환한 뒤 실행하는 구조입니다.

## 파일 구조

```text
config/base.json                 모든 전략 설정
src/team7_quant/
  module1.py                     시점별 유니버스와 제외 사유
  module2.py                     팩터 산식·percentile·CoreScore·IC
  module34.py                    빈티지 경기국면·7년 가중치표
  io.py                          Excel/CSV 구조 검사와 공통 검증
  config.py                      설정 검증
  fixtures.py                    기능시험용 합성자료
  cli.py                         명령행 실행기
templates/                       원본을 정규화할 CSV 양식
tests/                           미래정보·경계값·가중치 단위시험
```

## 투자규칙 충돌 처리

팀 명세에는 두 규칙이 공존하므로 코드가 임의로 섞지 않습니다.

| 설정 프로필 | CoreScore | 모멘텀 |
|---|---|---|
| `weighted_momentum` (기본) | 가치+퀄리티 80%, 모멘텀 20% | 가중합 |
| `momentum_filter` | 가치+퀄리티 100% | 하위 20% 제외 |

기본 프로필에서 모듈 4가 가치:퀄리티를 0.6:0.4로 판단하면 모듈 2에 전달되는 최종 가중치는
`0.48:0.32:0.20`입니다. 정적 중립은 `0.40:0.40:0.20`입니다.

## 설치와 실행

Windows PowerShell 또는 VS Code 터미널에서 프로젝트 폴더로 이동합니다.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -e .
python -m team7_quant inspect --input data/raw --out results/input_inspection.json
python -m team7_quant make-fixture --out examples/synthetic_bundle
python -m team7_quant demo --input examples/synthetic_bundle --config config/base.json --out results/synthetic
python -m unittest discover -s tests -v
```

`inspect`는 실제 Excel/CSV의 시트명·헤더·행 수만 확인합니다. DataGuide 항목명이나 단위를
자동 추측하지 않으므로, 원본명→표준열 대응표를 확정한 뒤 `templates/` 형식으로 정규화해야 합니다.
합성자료 출력은 코드 기능시험일 뿐 실제 백테스트 성과가 아닙니다.

## 입력 파일

### `security_master.csv`

```text
stock_code,listing_date,delisting_date,market_type,security_type,share_class
```

현재 상장종목은 `delisting_date`를 비워 둡니다. 미래에 상장폐지된 기업도 과거 상장기간에는
포함됩니다. 시장 이전 이력이 있으면 `effective_from`, `effective_to` 열을 추가할 수 있습니다.

### `prices.csv`

```text
date,stock_code,close,adj_close,trading_volume,trading_value,market_cap
```

유동성은 직전 63개 관측 거래일의 `trading_value` 평균으로 계산합니다. 모멘텀은 `adj_close`,
가치지표는 DataGuide 원자료 `market_cap`을 사용합니다.

### `financials.csv`

```text
stock_code,fiscal_year,fiscal_month_end,ni_parent,equity_parent,cfo,gross_profit,total_assets,total_liabilities,statement_type,available_at
```

`available_at`이 있으면 실제 사용 가능일을 우선합니다. 없으면 결산일+4개월 후 월말을 사용하며
`assumed_lag`로 기록합니다. 같은 결산기에 연결·별도가 모두 있으면 연결을 우선합니다.

### `sectors.csv`

```text
stock_code,date,sector_code,is_financial
```

금융업은 GPA를 결측 처리하고 나머지 퀄리티 지표로 평균을 냅니다.

### `macro.csv`

```text
observation_month,release_date,vintage_date,available_at,indicator_code,value,source,vintage_status
```

월말 판단일 이전에 공개된 버전만 사용합니다. 최신 수정자료밖에 없으면
`vintage_status=FINAL_VINTAGE_PROXY`로 표시합니다.

### `factor_returns.csv`

```text
formation_date,return_month,return_available_at,value_return,quality_return,return_basis,portfolio_rule_id
```

가치·퀄리티 단독 포트폴리오는 유니버스·종목 수·비중·모멘텀 규칙 등 다른 조건을 동일하게
맞춰야 합니다. 동적 전략의 수익률을 가중치 학습자료로 사용하지 않습니다.

## 주요 출력

| 파일 | 내용 |
|---|---|
| `module1_universe.csv` | 시점별 최종 유니버스 |
| `module1_audit.csv` | 전 종목 필터 값과 제외 사유 |
| `module2_scores.csv` | 원시지표·percentile·팩터점수 |
| `module2_selections.csv` | 상위 선정 종목 |
| `monthly_regimes.csv` | 관측·확정 국면과 전환 상태 |
| `annual_regime_weights.csv` | 연도별 7년 학습 가중치표 |
| `monthly_factor_weights.csv` | 모듈 2에 전달할 최종 가중치 |
| `training_samples.csv` | 포함·제외된 가중치 학습 표본 |

## 실제 자료 적용 전 확인사항

- 종목코드는 6자리 문자열로 유지합니다.
- 금액 단위를 통일하고 원본 단위를 기록합니다.
- 상장폐지 종목을 삭제하지 않습니다.
- 결측을 0이나 미래 값으로 채우지 않습니다.
- 공시일 이전 재무자료와 발표일 이전 경기자료를 사용하지 않습니다.
- 테스트 통과와 실제 전략의 성과 입증은 서로 다릅니다.
