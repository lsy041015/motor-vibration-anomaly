# 모터 진동 이상 탐지 (IC-PBL, 기계학습론)

모터 진동 신호(±2g 가속도, 약 401.9 Hz, 파일당 1024 샘플)로 정상과 불량 모터를 분류하는 프로젝트입니다.

- 평가 단위: 256 샘플 윈도우, 지표는 Test F1입니다.
- 분할: 모터 1~10번은 Test, 11~40번은 Training입니다(정상과 불량 모두). 겹침(0/25/50%)은 Training과 Test에 똑같이 적용합니다.
- 모델 선택: Training 모터만 쓰는 교차검증으로 합니다. Test 점수는 기록만 하고 선택에는 쓰지 않습니다.

## 최종 모델

| 항목 | 값 |
|---|---|
| 특징 | time(9) + harm(8) + fault(4) = 21개 |
| 검출기 | IsolationForest(300) + OneClassSVM(nu 0.01) + OneClassSVM(nu 0.05), 정상 윈도우만으로 학습 |
| 점수 | 검출기별 이상순위(Training 정상 대비)의 평균 |
| 임계값 | 0.972 (Training out-of-fold F1 최대) |
| Training CV F1 | 0.944 (속도 그룹 제외 CV 0.873, 불량 유형 제외 CV 0.914) |
| **Test F1** | **0.850** (TN 3823, FP 421, FN 770, TP 3366) |

## 방법

1. **회전주파수 f0 추정**: 15~60 Hz에서 가장 큰 피크를 찾고 포물선 보간으로 보정합니다(Hann 창, 1024점 zero-padding).
2. **harm 특징**: 1~6차 고조파 에너지 비율, 고조파 합, 비고조파 비율입니다.
3. **fault 특징**: 고정 불량 대역(55~65, 80~92, 100~115, 180~190 Hz)에서 k·f0 ±2.5 Hz를 뺀 에너지입니다.
   - 회전속도가 드문 정상 모터는 고조파가 불량 대역과 겹쳐 오탐이 났습니다. 이 특징이 그 문제를 줄입니다.
4. **선택용 CV 세 가지**(모두 Training만 사용):
   - cv: 모터 단위 GroupKFold(5)
   - spd: 회전속도 그룹 하나씩 제외
   - flt: 불량 유형 하나씩 제외
   - 규칙: cv F1 1위와 0.002 이내인 후보는 spd F1로 순위를 정합니다.

자세한 과정과 결과는 [`results/4단계_정리.md`](results/4단계_정리.md)에 있습니다.

## 한계

Test 불량 7번과 10번은 Training에 없는 불량 유형이라 많이 놓칩니다(정답률 51%, 0%). Training만으로는 이를 미리 알 수 없었습니다.

## 뷰어

[`viewer.html`](viewer.html)을 브라우저로 열면 됩니다. Plotly를 CDN에서 불러오므로 인터넷 연결이 필요합니다.

- **성능 탭**: 지표, 혼동행렬, 모터별 정답률, 이상점수 분포, 임계값 실험, 후보 모델 비교
- **분류·추론 과정 탭**: 윈도우 하나의 판정 과정(파형 → f0 → 특징 → 검출기별 순위 → 평균 → 판정)

## 실행

원본 데이터는 과목 제공 자료라 저장소에 넣지 않았습니다. `data/normal/<모터>/time_vib/*.csv`, `data/abnormal/<모터>/time_vib/*.csv` 구조로 두면 됩니다.

```bash
pip install -r requirements.txt
cd code
python tune.py 0        # 후보 탐색과 선택 -> results/tune4.csv
python viewer.py        # 1위 모델로 viewer.html 생성
```

| 파일 | 내용 |
|---|---|
| `code/load.py` | CSV 로드, 윈도우 분할, Training/Test 분할 |
| `code/features.py` | time, band, fft, harm, fault, order 특징 |
| `code/anomaly.py` | 이상탐지 검출기와 순위 변환 |
| `code/tune.py` | 특징 × 검출기 조합 탐색, 세 가지 CV, 선택 규칙 |
| `code/diagnose.py` | Training 오탐 원인 진단 |
| `code/supervised.py` | 지도학습 비교(LogReg, HistGB) |
| `code/side.py` | 보조 실험(파일 단위 평균, 시드 분산) |
| `code/cnn.py` | 1D CNN 실험(3단계) |
| `code/viewer.py` | 뷰어 생성 |
