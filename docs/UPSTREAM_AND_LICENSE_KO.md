[English](UPSTREAM_AND_LICENSE.md) | [한국어](UPSTREAM_AND_LICENSE_KO.md)

# Upstream과 라이선스

이 저장소는 [lab-emi/OpenDPD](https://github.com/lab-emi/OpenDPD)의 정식
GitHub fork이며, `98c2622cfef9ef537a250e118d811a6ae7333f53` 커밋에서 분기한
수정 저작물이다.
기존 Apache License 2.0 전문과 upstream 저자 표시를 유지한다. 논문 citation은
[고정한 upstream commit의 README](https://github.com/lab-emi/OpenDPD/blob/98c2622cfef9ef537a250e118d811a6ae7333f53/README.md)에서
확인할 수 있다.

주요 수정 범위는 다음과 같다.

- causal FExLite TCN과 구조 metadata
- Conv1d full-I/O QAT 및 train-only calibration
- PA, FP32 DPD, QAT 학습 산출물의 명시적 원자 출판
- exact-zero DGRU PA 안정화
- portable integer RTL export와 verifier
- full-test integer DPD + frozen PA evaluator
- checkpoint-independent regression test

기존 OpenDPD의 dataset 및 제3자 도구는 각각의 배포 조건을 따른다. 학습 checkpoint는
이 frontend Python package와 wheel에 포함하지 않는다. DPD-Flow monorepo 루트의
`artifacts/checkpoints/`에는 재현 검증용 소형 reference checkpoint가 별도 tracked
artifact로 포함될 수 있다. 합성 library와 EDA binary는 소스 배포물에 포함하지 않는다.

GitHub fork parent, 공식 upstream URL과 source commit이 provenance 기록을 이룬다.
`tcn-qat` branch가 DPD-Flow 수정사항을 소유한다. Upstream 변경은 전용 sync branch에서
`lab-emi/OpenDPD`로부터 fetch한 뒤 `tcn-qat`에 병합한다. DPD-Flow submodule pointer를
갱신하기 전에 local QAT, integer export, frozen-PA, dataset 및 regression 계약을 모두
보존하고 다시 검증해야 한다.
