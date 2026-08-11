[English](README_TCN_QAT.md) | [한국어](README_TCN_QAT_KO.md)

# 번들 PA 데이터셋 카탈로그

이 디렉터리에는 OpenDPD-TCN-QAT 프런트엔드와 함께 배포하는 PA 입력/출력 fixture가
있다. 측정 입력과 출력의 각 행은 시간축으로 정렬된 complex-baseband I/Q sample이다.
CSV 값은 데이터 경계에서 floating point로 유지하며, QAT 모델이 load 이후 설정된
fake quantization을 적용한다.

체크인된 DPD-Flow 설정이 고정해서 사용하는 데이터셋은 `DPA_200MHz`다. 다른
데이터셋은 프런트엔드 실험 범위를 넓히고 다른 저장 및 복조 경로를 검증하기 위한
것이다. 번들에 포함됐다는 사실만으로 DPD-Flow PPA 또는 DPD 품질의 검증된
baseline이 되는 것은 아니다.

## 디렉터리 구성

```text
datasets/
├── APA_200MHz/          # Split CSV, CP가 있는 LTE/OFDM, 측정 A
├── APA_200MHz_b/        # Split CSV, CP가 있는 LTE/OFDM, 측정 B
├── DPA_160MHz/          # Split CSV, CP 없는 IFFT frame
├── DPA_200MHz/          # Split CSV, 검증된 DPD-Flow fixture
├── MyCustomPA/          # Single-CSV custom dataset 예제
├── MATLAB/
│   └── signal_generation/
│       ├── iterative_match.py
│       └── test_iterative_match.py
├── demodulator.py       # 공통 demodulator base class와 factory
└── plot_utils.py        # 공통 진단 plot 생성기
```

각 PA dataset 디렉터리에는 `spec.json`, dataset별 `demod.py`,
`plot_dataset.py` wrapper가 있다. 진단 PNG는 plotting wrapper로 재생성할 수 있는
산출물이며 필수 입력이나 dataset format의 일부가 아니다.

## 데이터 형식과 API 경로

### Split CSV 형식

`APA_200MHz`, `APA_200MHz_b`, `DPA_160MHz`, `DPA_200MHz`는 각각 다음 파일
6개를 사용한다.

```text
train_input.csv   train_output.csv
val_input.csv     val_output.csv
test_input.csv    test_output.csv
```

모든 파일의 header는 `I,Q`다. 서로 대응하는 input과 output 파일의 행 수는 같고
sample 단위로 정렬돼 있다. Split 파일은 이미 물리적으로 나뉘어 있으므로 재현 실험
중에 합친 뒤 무작위로 다시 분할하면 안 된다.

### Single CSV 형식

`MyCustomPA/data.csv`는 `I_in,Q_in,I_out,Q_out` column 4개를 가진다.
`spec.json`은 `dataset_format`을 `single_csv`로 선언하고 `data.csv` 파일명과 순서가
유지되는 split 경계를 기록한다. Loader는 shuffle 없이 `[0:58982)`를 train,
`[58982:78642)`를 validation, `[78642:98304)`를 test로 자른다.

### Public API 경로

다음 예제는 `frontend/OpenDPD-TCN-QAT/`에서 실행한다. Public
`opendpd.load_dataset()` 함수는 filesystem 경로를 받아 NumPy array dictionary를
반환한다.

```python
import opendpd

data = opendpd.load_dataset("datasets/DPA_200MHz")
print(data["X_train"].shape, data["y_test"].shape)
```

반면 학습 및 추론 API는 번들 데이터를 디렉터리 이름으로 해석한다.

```python
import opendpd

pa_result = opendpd.train_pa(
    dataset_name="DPA_200MHz",
    PA_backbone="gru",
    PA_hidden_size=23,
    n_epochs=100,
    accelerator="cuda",
)
```

Dataset별 demodulator는 공통 factory로 선택한다.

```python
from datasets.demodulator import Demodulator

demodulator = Demodulator.from_dataset("APA_200MHz")
```

## 데이터셋 카탈로그

| 항목 | 형식 | 실제 train / val / test sample | 신호 및 demodulator | 용도 |
|---|---|---:|---|---|
| `APA_200MHz` | `split_csv` | 58,980 / 19,662 / 19,662 | 5-carrier LTE TM3.1a, CP-aware `OFDMCPDemodulator` | APA 측정 A, CP 동기화·equalization·표준 OFDM 실험 |
| `APA_200MHz_b` | `split_csv` | 58,980 / 19,662 / 19,662 | 측정 A와 같은 waveform class 및 demodulator | 독립 측정 B, capture 간 비교 |
| `DPA_160MHz` | `split_csv` | 294,912 / 98,304 / 98,304 | 4-carrier CP 없는 IFFT frame, `IFFTFrameDemodulator` | 큰 frame, 1024QAM, 160 MHz DPA 실험 |
| `DPA_200MHz` | `split_csv` | 23,040 / 7,680 / 7,680 | 10-carrier CP 없는 IFFT frame, `IFFTFrameDemodulator` | 체크인된 DPD-Flow 학습·QAT·정수·RTL 비교 baseline |
| `MyCustomPA` | `single_csv` | 58,982 / 19,660 / 19,662 | Single-channel 예제, `IFFTFrameDemodulator` | Custom single-CSV import 계약 예제, 검증 baseline은 아님 |
| `MATLAB/signal_generation` | Python helper, PA CSV split 없음 | 해당 없음 | 고정 5-carrier MATLAB reference signal matcher | 신호 생성 정합 연구, PA/DPD 학습에서는 사용하지 않음 |

위 수치는 실제 CSV 행 수다. Split dataset 4개는 `spec.json`에도 명목상
`0.6/0.2/0.2` 비율을 선언한다. 정수 반올림으로 차이가 있으면 실제 파일을 기준으로
삼는다.

### APA_200MHz와 APA_200MHz_b

두 APA 항목은 모두 5-carrier LTE TM3.1a waveform sample 98,304개를 포함한다. 각
carrier의 점유 bandwidth는 20 MHz이고 carrier center 간격은 40 MHz다. 신호는
491.52 MHz와 15 kHz subcarrier spacing으로 생성됐으며, dataset metadata에는
983.04 MHz에서 유효 spacing이 30 kHz인 것으로 기록돼 있다. Composite occupied
bandwidth는 200 MHz이고 data modulation은 256QAM이다.

두 `spec.json`은 같은 신호 parameter를 사용한다.

| Field | 값 |
|---|---:|
| `input_signal_fs` | 983.04 MHz |
| `bw_main_ch` | 200 MHz |
| `bw_sub_ch` | 40 MHz carrier spacing |
| `n_sub_ch` | 5 |
| `nperseg` | 19,662 sample |
| `ofdm_nfft` | 32,768 sample |
| `n_active` | Carrier당 subcarrier 600개 |
| `cp_first` / `cp_other` | 2,560 / 2,304 sample |
| `scs` | 유효 30 kHz |
| `papr_db` | 10.0 dB |

APA data에서 `nperseg`는 PSD/평가 segment 길이지 OFDM FFT frame이 아니다.
`OFDMCPDemodulator`는 각 carrier를 분리하고 cyclic-prefix correlation으로 symbol
경계를 찾은 뒤 FFT offset을 미세 조정하고 active subcarrier를 추출한다. 또한 깨끗한
입력 reference를 기준으로 PA 출력을 equalize할 수 있다. 짧은 validation 및 test
split은 각각 32,768-sample FFT symbol과 CP 전체를 포함하지 않으므로, plotting code는
full-sequence APA constellation이 필요할 때 전체 연결 sequence를 사용한다.

`APA_200MHz_b`는 같은 waveform class를 사용한 두 번째 capture다. 별도 dataset
이름을 유지해야 하며, 측정 A와 B 결과를 합치거나 하나의 split처럼 보고하면 안 된다.

### DPA_160MHz

`DPA_160MHz`는 640 MHz sample rate의 4-carrier, 160 MHz DPA waveform sample
491,520개를 포함한다. 각 carrier는 40 MHz를 점유하고 `spec.json`에 기록된 modulation은
1024QAM이며, CP 없는 각 IFFT frame은 16,384 sample이다. 이 값으로
`IFFTFrameDemodulator`는 carrier당 active subcarrier 1,024개를 계산한다.

Demodulator는 신호를 정렬된 16,384-sample frame으로 나누고 frame마다 FFT를 한 번
적용한 뒤 네 carrier center 주변 bin을 직접 선택한다. 기본 DPA fixture보다 긴 temporal
segmentation, 큰 FFT frame, 높은 modulation order를 실험하는 데 적합하다.

### DPA_200MHz

`DPA_200MHz`는 800 MHz sample rate의 10-carrier, 200 MHz DPA waveform sample
38,400개를 포함한다. 각 carrier는 20 MHz를 점유하고 modulation은 64QAM이며, CP 없는
IFFT frame 크기는 `nperseg=2560`이다. 따라서 demodulator는 carrier당 active
subcarrier 64개를 계산한다.

CSV split 6개와 `spec.json`은 체크인된 H10/A12W12 및 H13/A14W14 flow 설정이
참조하는 측정 data fixture다. 검증된 segmented 평가는 2,560-sample 경계를 사용하고,
불완전한 마지막 segment의 오른쪽을 zero padding하며, 각 경계에서 integer TCN history와
고정 PA state를 reset한다. Continuous-stream 결과는 state semantics가 다르므로 별도로
표시해야 한다.

### MyCustomPA

`MyCustomPA`는 순서가 유지되는 row 98,304개로 `single_csv` 계약을 보여준다. 예제
metadata는 800 MHz sample rate에서 하나의 200 MHz channel과 `nperseg=2560`을
기술한다. `modulation` field는 없으므로 디렉터리 이름이나 sample 값으로 modulation
order를 추론하면 안 된다.

Public API를 사용해 4-column 측정 CSV로 새 dataset을 만든다.

```python
import opendpd

dataset_dir = opendpd.create_dataset(
    csv_path="/path/to/measurements.csv",
    output_dir="datasets",
    dataset_name="MyPA",
    dataset_format="single_csv",
    input_signal_fs=800e6,
    bw_main_ch=200e6,
    bw_sub_ch=200e6,
    n_sub_ch=1,
    nperseg=2560,
)
```

6-file 구성을 생성하려면 `dataset_format="split_csv"`를 사용한다. 새 dataset을
baseline으로 취급하기 전에 측정 provenance, split 경계, framing/reset protocol, 신호
metadata, 고정 PA checkpoint를 고정해야 한다.

### MATLAB 신호 생성 helper

`MATLAB/signal_generation/iterative_match.py`는 생성 waveform을 MATLAB `.mat`
reference에 맞추는 Python 연구 helper다. PA 측정 dataset이 아니며
train/validation/test split도 없고, `opendpd.load_dataset()` 또는 PA/DPD 학습 API가
읽지도 않는다.

이 helper는 491.52 MHz sample rate에서 100 MHz를 차지하는 20 MHz carrier 5개,
32,768-point FFT, active subcarrier 1,200개, normal CP 2,304 sample로 구성된 sample
98,304개에 특화돼 있다. Analytical warm start 후 differential evolution으로 보정하고
NMSE, PSD mean absolute error, carrier별 power error, PAPR 차이, CCDF deviation, EVM을
검사한다. `test_iterative_match.py`에는 unit 및 reference-signal 기대값이 기록돼 있다.

이 디렉터리는 독립적으로 지원하는 CLI가 아니라 source snapshot이다. Helper는 번들에
없는 companion module `generate_signal.py`, `plot_comparison.py`와 기본 target `.mat`
파일을 import한다. 실행하려면 원래 연구 환경에서 이 입력들을 제공해야 한다. 또한 해당
상수는 100 MHz, 491.52 MHz reference waveform을 기술하므로 위의 200 MHz APA dataset
metadata 대신 사용하면 안 된다.

## 신호 metadata와 복조

`spec.json`은 sample rate, bandwidth, carrier 수, segmentation, modulation, format의
machine-readable source다. 디렉터리 이름으로 이 값을 추론하면 안 된다.
`datasets.demodulator.Demodulator.from_dataset(name)`은 이 파일을 읽고
`datasets.<name>.demod`를 import한다.

구현된 waveform family는 두 가지다.

- `IFFTFrameDemodulator`: cyclic prefix 없이 정렬해 연속 배치한 IFFT frame.
  `nperseg`가 생성 IFFT 크기와 정확히 같아야 한다.
- `OFDMCPDemodulator`: CP가 있는 표준 OFDM. APA specification의 `ofdm_nfft`,
  `cp_first`, `cp_other`, `scs`, `n_active`를 사용한다.

Demodulation은 constellation 시각화와 EVM 해석에 영향을 준다. 모델 학습이 소비하는
time-domain sample 자체를 바꾸지는 않는다.

## 진단 plot 재생성

`frontend/OpenDPD-TCN-QAT/`에서 dataset wrapper 하나 또는 다섯 개 모두를 실행한다.

```bash
python3 datasets/APA_200MHz/plot_dataset.py
python3 datasets/APA_200MHz_b/plot_dataset.py
python3 datasets/DPA_160MHz/plot_dataset.py
python3 datasets/DPA_200MHz/plot_dataset.py
python3 datasets/MyCustomPA/plot_dataset.py
```

각 wrapper는 dataset 옆에 `waveform.png`, `psd.png`, `constellation.png`, `amam.png`,
`ampm.png`를 재생성할 수 있다. 이 PNG는 폐기 가능한 진단 산출물이다. Checkout에
포함되는지는 선택 사항이며 학습, 평가, RTL export가 의존하면 안 된다. Plot 생성은 CSV
측정값을 읽기만 하며 수정하지 않는다.
