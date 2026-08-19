[English](TCN_QAT_ARCHITECTURE.md) | [한국어](TCN_QAT_ARCHITECTURE_KO.md)

# TCN-QAT 구조

## 목적

OpenDPD upstream은 FP32 TCN 학습은 지원하지만 Conv1d 기반 TCN을 위한 native QAT,
raw/output I/Q 경계 양자화, RTL integer export를 하나의 검증 경로로 제공하지 않는다.
이 포크는 포함된 5개 dataset, frozen PA, 학습/metric infrastructure를
보존하면서 해당 경로를 추가한다.

## 모델

FExLite feature는 다음 순서다.

```text
I, Q, p=I²+Q², p², I·p, Q·p
```

네트워크는 6→H pointwise input projection, `L`개의 H-channel depthwise causal
convolution, H→2 output projection, raw I/Q residual로 구성된다. Temporal layer `i`의
dilation은 `dilation_base**i`다. H, L, K, dilation base는 constructor와 OpenDPD CLI
인자로 바꿀 수 있다.

각 temporal convolution은 왼쪽 causal context를 만들기 위해 PyTorch padding 뒤
오른쪽 padding을 `Chomp1d`로 제거한다. 미래 sample을 변경해도 이전 output이 바뀌지
않는 causality test를 포함한다.

## Checkpoint 구조 정보

모델은 persistent `_rtl_spec=[version,L,K,dilation_base]` buffer를 가진다. QAT
checkpoint가 이 값을 보존하므로 exporter가 layer 수와 dilation을 파일명 없이 복원한다.
기존 `_rtl_spec` 없는 canonical checkpoint는 convolution weight shape로 H/L/K를 읽고
legacy dilation base 2를 명시적으로 적용한다.

## 명시적 학습 입출력과 저장 이름

전체 재현 실행은 의존 순서대로 산출물 세 개를 명시적으로 출판한다.
`train_pa --pa_output_checkpoint`는 PA 대리 모델을 출판한다. 양자화하지 않은
`train_dpd`는 바로 그 PA를 `--pa_checkpoint`로 받고 FP32 DPD를
`--dpd_output_checkpoint`로 출판한다. QAT 실행은 두 산출물을 각각
`--pa_checkpoint`, `--pretrained_model`로 받고 새 QAT 결과를
`--qat_output_checkpoint`로 출판한다.

PA 입력 경로를 생략하면 기존 사용법과의 호환을 위해 종전
`save/<dataset>/train_pa/` 탐색 규칙을 사용할 수 있다. 결합 flow는 이 암묵적
탐색에 의존하지 않으며 runner 소유 checkpoint를 생성하거나 명시적인 불변
checkpoint를 전달한다.

학습 logger가 이번 invocation에서 best checkpoint를 한 번 이상 저장한 경우에만 PA,
FP32 DPD, QAT 결과를 출판한다. 출력과 같은 디렉터리의 임시 파일을 완전히 기록한 뒤
`os.replace`로 교체한다. QAT calibration 및 model-spec JSON도 같은 내용으로
원자적으로 기록한다. 이전 실행 파일이 남아 있어도 이번 실행에서 저장이 없으면 출판은
실패한다.

FExLite TCN의 내부 logger 파일명에는 H뿐 아니라 L, K, dilation base가 포함된다. QAT이면
A/W bit 수도 포함되어 topology 또는 precision이 다른 실험이 같은 내부 경로를 덮어쓰지
않는다. 비-TCN 이름 규칙은 바꾸지 않았고, inference 경로는 새 이름이 없을 때 종전 TCN
이름을 한 번 더 탐색한다.

## QAT 환경

`FExLiteTCNQuantEnv`는 GRU quantization 환경과 분리되어 있다. 모든 Conv1d를
`INT_Conv1D`로 바꾸며 weight와 activation quantizer는 layer마다 독립이다. Pretrained
bias를 복사하고, weight 최대값이 clip되지 않도록 signed code range를 덮는 가장 작은
power-of-two scale로 초기화한다.

Raw input quantizer는 FEx보다 앞에 있으며 최종 output quantizer는 residual 뒤에 있다.
A-bit activation code의 물리 I/Q 경계 scale은 `2^(1-A)`, zero point는 0이다. 즉 학습
dataset 파일은 FP32로 읽지만 model graph에 들어간 직후 선택한 signed grid로 fake
quantize된다.

Calibration은 train loader의 최초 N batch를 한 번 cache하고 모든 Conv1d가 동일 sample을
관찰하도록 순차 수행한다. Raw/output interface scale은 고정하고 내부 activation scale만
absolute quantile을 덮는 power-of-two로 설정한다.

Checkpoint는 fail-closed rounding-policy record도 보존한다. `baseline_rne`는 모든
runtime requantization에 RNE를 사용한다. `prehs_floor`는 HardSwish 전후 경계에 signed
floor를 사용하고 FEx와 최종 residual은 RNE를 유지한다. `global_floor`는 FEx feature와
residual/output 경계에도 signed floor를 적용한다. Raw ADC code 생성과 저장
weight/bias 양자화는 세 mode 모두 RNE다. Training, export, evaluation은 같은 정책을
선택해야 하며 exporter는 불일치를 거부한다.

## Export와 정합성 경계

Exporter는 `manifest.json`, `weights/*.mem`, `golden_vectors/*.mem`으로 구성된
self-contained `opendpd_fexlite_qat_rtl_export` v2 package를 출판한다. 모든 package
경로는 상대경로이며 manifest는 각 memory의 SHA-256 identity와 canonical
`opendpd_fexlite_rounding_policy`, capability SHA-256을 기록한다. Version 1은 baseline
RNE가 기본이며 명시적인 activation-boundary와 layer별 `hardswish_input` 근거가 있을
때 과거 `prehs_floor`로 추론한다. Version 1은 global floor를 표현할 수 없다.
TCN-Compiler는 이
package만 입력으로 받고 frontend의 Python module, checkpoint, dataset loader를 import하지
않는다.

Export가 정의하는 exact integer evaluator가 hardware reference다. Package를 다시 읽은
결과는 모든 integer golden trace와 0 LSB로 일치해야 하며 RTL도 독립적으로 같은 trace와
0 LSB로 일치해야 한다. Fake-QAT 실행은 별도의 정합성 수준이다. PyTorch operation과
quantizer 배치 때문에 작은 code 차이가 생길 수 있으므로 fake-QAT-versus-integer 결과는
integer-versus-RTL 정합성과 분리해 측정하고 보고한다.

`global_floor`는 topology와 precision별로 DPD-Flow seed-4 pilot 및 paired 5-seed
frozen-PA 게이트를 통과해야 한다. 실패 결과는 연구 근거로 보존하고 downstream RTL과
AutoPipe에는 검증된 fallback 계약을 선택한다.

Canonical rounding, saturation, FEx, tap order, HardSwish, residual 산술은 monorepo의
[수치 계약](https://github.com/chldjwls85/DPD-Flow/blob/main/docs/NUMERIC_CONTRACT_KO.md)
한 곳에서 관리한다. 서로 다른 세 가지
정합성 claim과 검증된 baseline 결과는
[검증 정책](https://github.com/chldjwls85/DPD-Flow/blob/main/flow/docs/VALIDATION_KO.md)을
참고한다.
