<p align="center">
  <img src="assets/motor-vibration-anomaly-banner.png" alt="모터 진동 이상 탐지 — 속도 인식 특징과 이상 탐지 앙상블 (IC-PBL)" width="100%">
</p>

# 모터 진동 이상 탐지 (IC-PBL, 기계학습론)

기계학습론 IC-PBL 과제로, 모터 진동 신호(±2g, 약 401.9 Hz)를 256 샘플 윈도우로 잘라 정상과 불량을 가려내는 프로젝트예요. 모터 1~10번이 Test, 11~40번이 Training이고, 모델은 Training 모터만 쓴 교차검증으로 골랐어요. Test 점수는 기록만 하고 선택에는 안 썼어요.

## 최종 모델

| 항목 | 값 |
|---|---|
| 특징 | time(9) + harm(8) + fault(4) = 21개 |
| 검출기 | IsolationForest(300) + OneClassSVM(nu 0.01) + OneClassSVM(nu 0.05), 정상 윈도우만으로 학습 |
| 점수 | 검출기별 이상순위(Training 정상 대비)의 평균 |
| 임계값 | 0.972 (Training out-of-fold F1 최대) |
| Training CV F1 | 0.944 (속도 그룹 제외 CV 0.873, 불량 유형 제외 CV 0.914) |
| **Test F1** | **0.850** (TN 3823, FP 421, FN 770, TP 3366) |

핵심은 회전속도였어요. 스펙트럼에서 회전주파수 f0를 먼저 찾고 고조파 특징(harm)을 뽑았는데, 회전속도가 드문 정상 모터는 고조파가 불량 대역에 겹쳐서 오탐이 났어요. 그래서 불량 대역 에너지에서 k·f0 근처를 빼는 fault 특징을 만들었더니 많이 줄었어요. 선택할 때는 모터 단위 CV 말고도 속도 그룹이나 불량 유형을 하나씩 빼는 CV를 같이 봤어요.

아쉬운 건 Test의 불량 7번과 10번이에요. Training에 없는 불량 유형이라 많이 놓쳐요(정답률 51%, 0%). 과정은 [`results/4단계_정리.md`](results/4단계_정리.md)에 자세히 적어 뒀어요.

## 실행

원본 데이터는 과목 자료라 올리지 않았어요. `data/normal/<모터>/time_vib/*.csv`, `data/abnormal/<모터>/time_vib/*.csv`로 두면 돼요.

```bash
pip install -r requirements.txt
cd code
python tune.py 0        # 후보 탐색과 선택 -> results/tune4.csv
python viewer.py        # 1위 모델로 viewer.html 생성
```

결과는 [`viewer.html`](viewer.html)을 브라우저로 열면 볼 수 있어요(Plotly를 CDN에서 불러와서 인터넷이 필요해요). 혼동행렬, 모터별 정답률, 임계값 실험, 그리고 윈도우 하나가 판정되는 과정을 단계별로 보여 줘요.
