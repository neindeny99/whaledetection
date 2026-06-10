# 🐋 고래 종 분류 페이지 (Whale Species Classifier)

CNN 기반 **34종 해양 포유류** 소리 분류 웹 페이지.
사용자가 업로드한 고래 소리에서 가장 유사한 종을 **1위부터 5위까지** 확률(%)로 보여줍니다.

## 🖥 화면 구성

본 앱은 두 개의 화면으로 구성됩니다:

1. **업로드 화면** — 오디오 파일 업로드 또는 마이크 녹음 후 "분류 시작" 클릭
2. **결과 화면** — 가장 닮은 고래 1위 강조 + Top-5 순위 + 멜 스펙트로그램 표시

"다른 소리 다시 분석하기" 버튼으로 다시 업로드 화면으로 돌아갈 수 있습니다.

## 🚀 실행 방법

```bash
pip install -r requirements.txt
# best_whale_model.h5 를 같은 폴더에 배치
python app.py
```

브라우저에서 `http://localhost:7860` 열기.
인터넷으로 공유하려면 `app.py` 마지막 줄의 `share=False` → `share=True`.

## 🐳 분류 가능한 34종 (알파벳 순)

```
 0. Atlantic Spotted Dolphin       17. Killer Whale (Orca)
 1. Beluga / White Whale           18. Long-Beaked (Pacific) Common Dolphin
 2. Bottlenose Dolphin             19. Long-Finned Pilot Whale
 3. Boutu / Amazon River Dolphin   20. Melon-Headed Whale
 4. Bowhead Whale                  21. Minke Whale
 5. Clymene Dolphin                22. Narwhal
 6. Common Dolphin                 23. Northern Right Whale
 7. Dall's Porpoise                24. Pantropical Spotted Dolphin
 8. Dusky Dolphin                  25. Rough-Toothed Dolphin
 9. False Killer Whale             26. Short-Finned (Pacific) Pilot Whale
10. Fin / Finback Whale            27. Southern Right Whale
11. Fraser's Dolphin               28. Sperm Whale
12. Grampus / Risso's Dolphin      29. Spinner Dolphin
13. Gray Whale                     30. Striped Dolphin
14. Harbor Porpoise                31. Tucuxi Dolphin
15. Heaviside's Dolphin            32. White-Beaked Dolphin
16. Humpback Whale                 33. White-Sided Dolphin
```

> 라벨은 첨부된 `confusion_matrix.png` 에서 직접 추출했습니다 (학습 시 폴더 알파벳 순서로 가정).

## ⚙️ 오디오 전처리 파라미터 (수정 필요 시 `app.py` 상단)

```python
SR       = 22050   # sample rate
N_MELS   = 40      # 모델 입력의 mel band 수
N_FRAMES = 50      # 모델 입력의 time frame 수
N_FFT    = 2048
HOP      = 512
```

학습 코드의 `librosa.feature.melspectrogram(...)` 호출 인자와 정확히 동일해야 합니다.
값이 다르면 정확도가 크게 떨어집니다.

## 📦 파일 구성

```
whale_app/
 ├── app.py              ← Gradio 메인 앱 (두 화면 전환)
 ├── requirements.txt
 ├── README.md
 └── best_whale_model.h5 ← 학습된 모델
```

## 🧠 모델 사양

- 아키텍처: Sequential CNN (Conv2D ×3 + Dense ×2)
- 입력: `(40, 50, 1)` log-mel spectrogram
- 출력: **34-class softmax**
- 학습 곡선(첨부): val accuracy ≈ 93%, val loss ≈ 0.30

## 📚 참고 문헌

- Suarez D. (2023). *Whale Detection through Sound Analysis with CNN*. Kaggle.
  https://www.kaggle.com/code/diegoasuarezg/whale-detection-through-sound-analysis-w-cnn
- 윤영글, 김한수, 김범규, 조성호, 강돈혁, 김선효 (2024). *CNN 모델을 이용한 수중음향 자료의 돌고래 휘슬음 자동 분류 연구*. 한국수산과학회지 57(6), 743-751.

## 🛠 트러블슈팅

| 증상 | 해결 |
|---|---|
| 모든 입력이 같은 클래스로 분류됨 | 전처리 파라미터(SR/HOP/정규화)가 학습과 다름. 학습 코드와 맞추세요. |
| `OSError: Unable to open file` | `best_whale_model.h5`가 `app.py`와 같은 폴더에 있는지 확인. |
| 모델 출력이 34가 아님 | 다른 모델일 가능성. `LABELS_EN` 길이를 모델 출력 수에 맞추세요. |
| 마이크가 작동 안 함 | 브라우저에서 마이크 권한 허용. HTTPS 또는 localhost에서만 작동. |
