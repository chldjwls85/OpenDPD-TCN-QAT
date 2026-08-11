[English](TCN_QAT_WORKFLOW.md) | [한국어](TCN_QAT_WORKFLOW_KO.md)

# OpenDPD-TCN-QAT

OpenDPD-TCN-QAT는 DPD-Flow를 위해 관리하는
[lab-emi/OpenDPD](https://github.com/lab-emi/OpenDPD)의 정식 GitHub fork다.
DPD-Flow는 이 저장소의 특정 commit을 Git submodule로 고정해 사용한다. OpenDPD의
측정 dataset, frozen PA, metric infrastructure를 보존하면서 RTL backend가 사용하는
causal FExLite TCN을 위한 native fail-closed quantization-aware training(QAT) 경로를
추가한다.

Power amplifier 모델은 frozen software surrogate로 유지한다. DPD 모델만 양자화하고
export한 뒤 RTL로 lowering한다.

## 이 포크의 추가 기능

- hidden width `H`, temporal depth `L`, kernel size `K`, dilation base 파라미터화
- FEx 이전 signed raw-I/Q fake quantization과 residual 이후 signed output quantization
- Conv1d별 독립 weight/activation quantizer와 train split 전용 power-of-two activation
  calibration
- 모호하지 않은 모델 복원을 위한 persistent checkpoint topology metadata
- weight, scale, causal delay, 수치 규칙, hash, layerwise golden vector를 포함하는
  versioned integer export
- exact integer verification과 같은 frozen DGRU PA를 이용한 full-test 평가
- frontend-neutral TCN-Compiler backend로 전달되는 manifest-only 경계

## 설치

Frontend만 개발할 때는 이 저장소를 직접 clone해 editable mode로 설치한다.

```bash
git clone https://github.com/chldjwls85/OpenDPD-TCN-QAT.git
cd OpenDPD-TCN-QAT
python3 -m pip install -e .
```

통합 환경은 DPD-Flow를 submodule까지 재귀적으로 clone한 뒤 DPD-Flow 루트에서 세
local package를 설치한다.

```bash
git clone --recurse-submodules https://github.com/chldjwls85/DPD-Flow.git
cd DPD-Flow
python3 -m pip install -e .
python3 -m pip install -e ./flow
python3 -m pip install -e ./frontend/OpenDPD-TCN-QAT
```

Frontend에는 Python 3.10 이상과 PyTorch 2.0 이상이 필요하다. 학습에는 CUDA 지원
PyTorch를 권장하며 export와 integer verification은 CPU에서도 실행할 수 있다.

## Canonical FExLite TCN flow

다음 frontend 명령은 이 저장소 루트 또는 DPD-Flow 내부의
`frontend/OpenDPD-TCN-QAT/`에서 실행한다.

통합 runner는 준비된 checkpoint 없이 OpenDPD 전체 학습 순서를 실행할 수 있다.
`train_pa`는 명시적인 PA 출력을 출판하고, floating-point `train_dpd`는 명시적인
FP32 DPD 출력을 출판하며, QAT는 runner가 소유한 바로 그 산출물을 사용한다. 재개
가능한 전체 경로는 monorepo 루트에서
`flow/configs/h13_a14w14_seed4_fulltrain.json`을 사용한다. 아래 명령은 마지막
QAT 단계만 직접 실행하는 예시다.

### 1. Quantization-aware training

다음 H13/A14W14 예시는 구조가 일치하는 FP32 DPD와 명시적인 frozen-PA checkpoint에서
시작한다.

```bash
python3 main.py --step train_dpd --dataset_name DPA_200MHz \
  --DPD_backbone fexlite_causal_tcn \
  --DPD_hidden_size 13 --DPD_num_layers 4 \
  --tcn_kernel_size 5 --tcn_dilation_base 2 \
  --quant --n_bits_a 14 --n_bits_w 14 \
  --quantize_hardswish_input \
  --activation_rounding discard_lsb_signed_floor \
  --pretrained_model /path/to/DPD_FP32.pt \
  --pa_checkpoint /path/to/PA.pt \
  --qat_output_checkpoint /path/to/DPD_QAT.pt \
  --n_epochs 200 --accelerator cuda
```

Dataset CSV 값은 FP32 측정값으로 읽는다. 모델 진입 직후 raw I/Q를 fake quantize하므로
A14에는 외부 activation 경계도 포함된다. 원본 CSV를 integer 파일로 다시 쓰지는 않는다.
Calibration에는 train split만 사용한다. 출판된 checkpoint 옆에는 calibration 및
model-spec JSON sidecar가 함께 생성된다.

위 두 activation option을 사용하면 모든 넓은 MAC/bias accumulator를 HardSwish 전에
signed A14로 fake quantize하고, HardSwish 결과도 다음 convolution 입력에서 다시 fake
quantize한다. 두 내부 경계는 2의 보수 LSB를 버리므로 나누어떨어지지 않는 음수는
마이너스 무한대 방향으로 내려간다. Raw I/Q, FEx, 마지막 residual/output 경계는 RNE
계약을 유지하므로 이 실험은 activation datapath만 바꾼다.

### 2. Integer export

Export 디렉터리는 새 경로여야 하며 exporter는 기존 package를 덮어쓰지 않는다.

```bash
python3 scripts/export_fexlite_qat_rtl.py \
  --checkpoint /path/to/DPD_QAT.pt \
  --pa-checkpoint /path/to/PA.pt \
  --dataset-name DPA_200MHz \
  --input datasets/DPA_200MHz/test_input.csv \
  --output-dir /path/to/rtl_export
```

Portable package는 `manifest.json`, `weights/*.mem`, `golden_vectors/*.mem`으로
구성된다. TCN-Compiler는 이 package만 입력으로 받고 PyTorch module, checkpoint, dataset
loader를 직접 import하지 않는다.

### 3. Integer verification

```bash
python3 scripts/verify_fexlite_qat_rtl.py \
  --manifest /path/to/rtl_export/manifest.json
```

Verifier는 export된 memory를 다시 읽고 모든 integer golden trace와 0-LSB 정합을
요구한다.

### 4. Frozen-PA 평가

```bash
python3 scripts/evaluate_fexlite_integer_pa.py \
  --manifest /path/to/rtl_export/manifest.json \
  --pa-checkpoint /path/to/PA.pt \
  --qat-checkpoint /path/to/DPD_QAT.pt \
  --dataset-name DPA_200MHz --split test --protocol segmented \
  --device cuda --output /path/to/integer_pa_metrics.json
```

Segmented protocol은 각 `nperseg` 경계에서 TCN history와 frozen-PA state를 reset한다.
Integer-DPD RF metric을 보고하며 `--qat-checkpoint`를 주면 fake-QAT와 integer의
차이도 별도 항목으로 평가한다.

재개 가능한 전체 PA-modeling-to-synthesis 실행에는 monorepo 루트의 `dpdflow`를 사용한다.
[DPD-Flow 개요](https://github.com/chldjwls85/DPD-Flow/blob/main/README_KO.md)와
[통합 flow](https://github.com/chldjwls85/DPD-Flow/blob/main/flow/README_KO.md)를 참고한다.

## 포함된 dataset

이 fork에는 upstream OpenDPD의 `APA_200MHz`, `APA_200MHz_b`, `DPA_160MHz`,
`DPA_200MHz`, `MyCustomPA` 등 5개 dataset을 포함한다. 앞의 4개 측정 PA dataset은
train/validation/test split CSV를 사용하고, `MyCustomPA`는 단일 CSV custom-dataset
fixture다. 재생성 가능한 진단 PNG는 의도적으로 Git에서 제외하며 각 dataset의
`plot_dataset.py`로 다시 만들 수 있다. `DPA_200MHz`가 검증된 DPD-Flow
baseline이라는 점은 그대로이며, 다른 dataset의 포함 자체가 그 checkpoint나 평가
protocol을 자동으로 검증하지는 않는다. Data나 protocol을 바꾸기 전에
[dataset 계약](../datasets/README_TCN_QAT_KO.md)을 확인한다.

## 문서

- [TCN-QAT 구조와 export 경계](TCN_QAT_ARCHITECTURE_KO.md)
- [TCN-Compiler 수치 계약](https://github.com/chldjwls85/DPD-Flow/blob/main/docs/NUMERIC_CONTRACT_KO.md)
- [Flow 검증 정책](https://github.com/chldjwls85/DPD-Flow/blob/main/flow/docs/VALIDATION_KO.md)
- [Upstream provenance와 라이선스](UPSTREAM_AND_LICENSE_KO.md)

`examples/api_usage_example.py`는 여러 모델을 학습하고 custom dataset을 생성하는
state-changing generic upstream API demo다. Canonical FExLite TCN QAT 또는 DPD-Flow
workflow가 아니다.

## Upstream 표시

이 frontend는 [lab-emi/OpenDPD](https://github.com/lab-emi/OpenDPD)에서 파생한
수정 저작물이다. Upstream 저자 표시와 Apache License 2.0 전문을 보존하며, 논문
citation은 고정한 upstream README에서 확인할 수 있다. 정확한 원본 commit과 포크 변경 목록은
[provenance 문서](UPSTREAM_AND_LICENSE_KO.md)에 기록한다.
