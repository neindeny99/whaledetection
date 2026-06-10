"""
🐋 고래 종 분류 페이지 (Whale Species Classifier)
─────────────────────────────────────────────────
화면 구성:
  · 화면 1: 사용자가 고래 소리를 업로드하는 화면
  · 화면 2: 가장 유사한 고래 종 1~5위를 % 로 보여주는 결과 화면

모델: best_whale_model.h5  (Sequential CNN, input 40×50×1, output 36-class softmax)
라벨: 사용자가 첨부한 confusion_matrix.png 에서 추출한 34종 (알파벳 순)
참고: https://www.kaggle.com/code/diegoasuarezg/whale-detection-through-sound-analysis-w-cnn
"""

import os, io
import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
import gradio as gr

# ─────────────────────────────────────────────────────────────
# 1) 모델 로드
# ─────────────────────────────────────────────────────────────
MODEL_PATH = os.environ.get("WHALE_MODEL_PATH", "best_whale_model.h5")
print(f"[INFO] Loading model from {MODEL_PATH} ...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print(f"[INFO] Model loaded. Input: {model.input_shape}, Output: {model.output_shape}")

# ─────────────────────────────────────────────────────────────
# 2) 클래스 라벨 (혼동행렬에서 직접 추출, 알파벳 순)
#    모델 출력: 34 classes — 혼동행렬과 정확히 일치
# ─────────────────────────────────────────────────────────────
LABELS_EN = [
    "Atlantic Spotted Dolphin",                # 0
    "Beluga / White Whale",                    # 1
    "Bottlenose Dolphin",                      # 2
    "Boutu / Amazon River Dolphin",            # 3
    "Bowhead Whale",                           # 4
    "Clymene Dolphin",                         # 5
    "Common Dolphin",                          # 6
    "Dall's Porpoise",                         # 7
    "Dusky Dolphin",                           # 8
    "False Killer Whale",                      # 9
    "Fin / Finback Whale",                     # 10
    "Fraser's Dolphin",                        # 11
    "Grampus / Risso's Dolphin",               # 12
    "Gray Whale",                              # 13
    "Harbor Porpoise",                         # 14
    "Heaviside's Dolphin",                     # 15
    "Humpback Whale",                          # 16
    "Killer Whale (Orca)",                     # 17
    "Long-Beaked (Pacific) Common Dolphin",    # 18
    "Long-Finned Pilot Whale",                 # 19
    "Melon-Headed Whale",                      # 20
    "Minke Whale",                             # 21
    "Narwhal",                                 # 22
    "Northern Right Whale",                    # 23
    "Pantropical Spotted Dolphin",             # 24
    "Rough-Toothed Dolphin",                   # 25
    "Short-Finned (Pacific) Pilot Whale",      # 26
    "Southern Right Whale",                    # 27
    "Sperm Whale",                             # 28
    "Spinner Dolphin",                         # 29
    "Striped Dolphin",                         # 30
    "Tucuxi Dolphin",                          # 31
    "White-Beaked Dolphin",                    # 32
    "White-Sided Dolphin",                     # 33
]

LABELS_KO = [
    "대서양 알락돌고래",       # 0
    "흰돌고래(벨루가)",        # 1
    "큰돌고래",               # 2
    "아마존강돌고래",          # 3
    "북극고래",               # 4
    "클라이메네돌고래",        # 5
    "참돌고래",               # 6
    "까치돌고래",             # 7
    "더스키돌고래",           # 8
    "흑범고래",               # 9
    "긴수염고래",             # 10
    "프레이저돌고래",          # 11
    "리소돌고래",             # 12
    "귀신고래",               # 13
    "쇠돌고래",               # 14
    "헤비사이드돌고래",        # 15
    "혹등고래",               # 16
    "범고래",                 # 17
    "긴부리참돌고래",          # 18
    "참거두고래",             # 19
    "들쇠고래",               # 20
    "밍크고래",               # 21
    "일각돌고래",             # 22
    "북대서양참고래",          # 23
    "범열대알락돌고래",        # 24
    "들쭉돌고래",             # 25
    "태평양들쇠고래",          # 26
    "남방참고래",             # 27
    "향고래",                 # 28
    "스피너돌고래",           # 29
    "줄무늬돌고래",           # 30
    "투쿠시돌고래",           # 31
    "흰부리돌고래",           # 32
    "흰줄무늬돌고래",         # 33
]
N_CLASSES = model.output_shape[-1]
assert len(LABELS_EN) == len(LABELS_KO) == N_CLASSES, \
    f"라벨 개수({len(LABELS_EN)})와 모델 출력 클래스 수({N_CLASSES})가 다릅니다."

# ─────────────────────────────────────────────────────────────
# 3) 오디오 → 모델 입력 전처리  (MFCC)
#    ★ 학습 코드(whale_37_classifier4.py) 의 extract_mfcc() 와 동일 ★
# ─────────────────────────────────────────────────────────────
SR       = 22050
N_MFCC   = 40       # 학습 코드 N_MFCC
N_FRAMES = 50       # 학습 코드 MAX_LEN

def audio_to_mfcc(audio_path: str):
    """학습 코드와 100% 동일한 MFCC 전처리.

    - librosa.feature.mfcc 인자는 학습 코드처럼 sr/n_mfcc 만 명시 (그 외 기본값)
    - 시간 축: 리사이즈가 아니라 zero-padding / truncation 으로 50 프레임 맞춤
    - 정규화: z-score  (x - mean) / (std + 1e-8)
    """
    y, _ = librosa.load(audio_path, sr=SR)
    if len(y) == 0:
        raise ValueError("빈 오디오 파일입니다.")

    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC)          # (40, T)

    # 시간 축을 50 으로 통일
    if mfcc.shape[1] < N_FRAMES:
        pad_width = N_FRAMES - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode="constant")
    else:
        mfcc = mfcc[:, :N_FRAMES]                                    # (40, 50)

    # z-score 정규화 (학습 코드와 동일)
    mfcc_norm = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-8)

    return mfcc_norm[..., np.newaxis], mfcc  # 모델 입력용 / 시각화용

def make_spectrogram_image(mfcc: np.ndarray) -> Image.Image:
    """MFCC 히트맵을 다크 테마 이미지로 렌더링."""
    fig, ax = plt.subplots(figsize=(7.5, 2.8), dpi=110, facecolor="#0a1a2e")
    librosa.display.specshow(mfcc, sr=SR, x_axis="time", cmap="magma", ax=ax)
    ax.set_facecolor("#0a1a2e")
    ax.tick_params(colors="#9ec5dd", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#3a5f7d")
    ax.set_xlabel("Time (s)", color="#9ec5dd", fontsize=9)
    ax.set_ylabel("MFCC coefficient", color="#9ec5dd", fontsize=9)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

# ─────────────────────────────────────────────────────────────
# 4) 추론 + top-5
# ─────────────────────────────────────────────────────────────
def predict_top5(audio_path):
    x, mfcc = audio_to_mfcc(audio_path)
    x = np.expand_dims(x, axis=0)
    probs = model.predict(x, verbose=0)[0]   # (34,)

    top5_idx = np.argsort(probs)[::-1][:5]
    top5 = [(int(i), float(probs[i])) for i in top5_idx]

    spec_img = make_spectrogram_image(mfcc)
    return top5, spec_img

# ─────────────────────────────────────────────────────────────
# 5) 두 화면(업로드 / 결과) 전환 핸들러
# ─────────────────────────────────────────────────────────────
def go_analyze(audio_path):
    """업로드 화면 → 결과 화면."""
    if audio_path is None:
        gr.Warning("먼저 고래 소리 파일을 업로드하거나 녹음해주세요 🐋")
        return (
            gr.update(visible=True),   # upload_screen
            gr.update(visible=False),  # result_screen
            *[gr.update() for _ in range(13)],  # 5 ranks × (label, bar, pct) + spec + summary - filled below
        )

    try:
        top5, spec_img = predict_top5(audio_path)
    except Exception as e:
        gr.Warning(f"오디오 처리 중 오류: {e}")
        return (gr.update(visible=True), gr.update(visible=False),
                *[gr.update() for _ in range(13)])

    # 각 순위별 UI 업데이트 값 만들기
    medals = ["🥇", "🥈", "🥉", "④", "⑤"]
    rank_html_updates = []
    for rank in range(5):
        idx, prob = top5[rank]
        pct = prob * 100
        bar_pct = max(2.0, pct)  # 너무 작으면 안 보여서 최소 2%
        html = f"""
        <div class="rank-card rank-{rank+1}">
          <div class="rank-medal">{medals[rank]}</div>
          <div class="rank-body">
            <div class="rank-line">
              <span class="rank-num">{rank+1}위</span>
              <span class="rank-name">{LABELS_EN[idx]}</span>
              <span class="rank-pct">{pct:.2f}%</span>
            </div>
            <div class="rank-name-ko">{LABELS_KO[idx]}</div>
            <div class="bar-track">
              <div class="bar-fill" style="width:{bar_pct:.2f}%"></div>
            </div>
          </div>
        </div>
        """
        rank_html_updates.append(html)

    # 1등 강조용 큰 텍스트
    top_idx, top_prob = top5[0]
    hero_html = f"""
    <div class="hero-result">
      <div class="hero-label">가장 닮은 고래</div>
      <div class="hero-name">🐋 {LABELS_EN[top_idx]}</div>
      <div class="hero-name-ko">{LABELS_KO[top_idx]}</div>
      <div class="hero-pct">{top_prob*100:.2f}%</div>
    </div>
    """

    return (
        gr.update(visible=False),  # upload_screen 숨김
        gr.update(visible=True),   # result_screen 표시
        hero_html,                 # hero_box
        rank_html_updates[0],      # rank1_html
        rank_html_updates[1],      # rank2_html
        rank_html_updates[2],      # rank3_html
        rank_html_updates[3],      # rank4_html
        rank_html_updates[4],      # rank5_html
        spec_img,                  # spectrogram
    )

def go_back():
    """결과 화면 → 업로드 화면."""
    return gr.update(visible=True), gr.update(visible=False)

# ─────────────────────────────────────────────────────────────
# 6) UI (Gradio Blocks)
# ─────────────────────────────────────────────────────────────
CUSTOM_CSS = """
:root {
  --deep-1: #050d1a;
  --deep-2: #0a1a2e;
  --deep-3: #102844;
  --accent: #4ec5d8;
  --accent-2: #f5c451;
  --text-1: #eef6fb;
  --text-2: #9ec5dd;
}

.gradio-container, .gradio-container * {
  font-family: 'Pretendard','Inter',-apple-system,sans-serif !important;
}

.gradio-container {
  background:
    radial-gradient(1100px 600px at 15% -10%, #1b4368 0%, transparent 55%),
    radial-gradient(900px 500px at 95% 110%, #102844 0%, transparent 60%),
    var(--deep-1) !important;
  color: var(--text-1) !important;
  min-height: 100vh;
}

/* ─── 공통 ─── */
.screen-wrap { max-width: 880px; margin: 0 auto; padding: 32px 16px 48px; }
.brand {
  text-align: center; margin-bottom: 28px;
}
.brand-title {
  font-size: 2rem; font-weight: 800; letter-spacing: -0.02em;
  background: linear-gradient(90deg, #fff 0%, var(--accent) 55%, var(--accent-2) 100%);
  -webkit-background-clip: text; background-clip: text; color: transparent;
  margin: 0;
}
.brand-sub {
  color: var(--text-2); font-size: 0.92rem; margin-top: 6px;
  letter-spacing: 0.02em;
}

/* ─── 화면 1: 업로드 ─── */
.upload-card {
  background: linear-gradient(180deg, rgba(20,40,68,0.7), rgba(10,26,46,0.7));
  border: 1px solid rgba(78,197,216,0.18);
  border-radius: 22px;
  padding: 36px 32px;
  box-shadow: 0 24px 60px -20px rgba(0,0,0,0.5);
  backdrop-filter: blur(6px);
}
.upload-title {
  text-align: center; font-size: 1.4rem; font-weight: 700;
  color: var(--text-1); margin: 0 0 6px;
}
.upload-desc {
  text-align: center; color: var(--text-2); font-size: 0.92rem;
  margin-bottom: 26px;
}
#analyze-btn {
  background: linear-gradient(135deg, #4ec5d8, #2d8aa0) !important;
  color: white !important; border: none !important;
  font-weight: 700 !important; font-size: 1.05rem !important;
  padding: 14px 28px !important; border-radius: 14px !important;
  box-shadow: 0 12px 30px -8px rgba(78,197,216,0.55) !important;
  transition: transform .15s ease, box-shadow .15s ease !important;
}
#analyze-btn:hover {
  transform: translateY(-2px);
  box-shadow: 0 18px 40px -8px rgba(78,197,216,0.7) !important;
}

/* ─── 화면 2: 결과 ─── */
.hero-result {
  text-align: center;
  padding: 28px 20px 24px;
  background: linear-gradient(180deg, rgba(78,197,216,0.10), rgba(245,196,81,0.04));
  border: 1px solid rgba(78,197,216,0.25);
  border-radius: 22px;
  margin-bottom: 22px;
}
.hero-label {
  color: var(--text-2); font-size: 0.85rem; letter-spacing: 0.18em;
  text-transform: uppercase; margin-bottom: 8px;
}
.hero-name {
  font-size: 1.85rem; font-weight: 800;
  color: var(--text-1); letter-spacing: -0.01em;
  margin: 4px 0 2px;
}
.hero-name-ko {
  font-size: 1.05rem; color: var(--accent); margin-bottom: 12px;
}
.hero-pct {
  font-size: 2.6rem; font-weight: 900;
  background: linear-gradient(90deg, var(--accent), var(--accent-2));
  -webkit-background-clip: text; background-clip: text; color: transparent;
}

.rank-list { display: flex; flex-direction: column; gap: 10px; margin-top: 8px; }
.rank-card {
  display: flex; align-items: center; gap: 14px;
  background: rgba(16,40,68,0.55);
  border: 1px solid rgba(78,197,216,0.12);
  border-radius: 14px;
  padding: 12px 16px;
}
.rank-card.rank-1 { border-color: rgba(245,196,81,0.45); background: rgba(245,196,81,0.06); }
.rank-card.rank-2 { border-color: rgba(199,210,219,0.30); }
.rank-card.rank-3 { border-color: rgba(205,127, 50,0.35); }
.rank-medal {
  font-size: 1.6rem; width: 38px; text-align: center; flex-shrink: 0;
}
.rank-body { flex: 1; min-width: 0; }
.rank-line {
  display: flex; align-items: baseline; gap: 10px;
  margin-bottom: 4px;
}
.rank-num {
  color: var(--text-2); font-size: 0.78rem; font-weight: 700;
  letter-spacing: 0.05em; flex-shrink: 0;
}
.rank-name {
  color: var(--text-1); font-weight: 700; font-size: 1rem;
  flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.rank-name-ko {
  color: var(--text-2); font-size: 0.82rem; margin-bottom: 6px;
}
.rank-pct {
  color: var(--accent); font-weight: 800; font-size: 1.05rem;
  flex-shrink: 0;
}
.rank-1 .rank-pct { color: var(--accent-2); }
.bar-track {
  height: 6px; width: 100%;
  background: rgba(255,255,255,0.06);
  border-radius: 999px; overflow: hidden;
}
.bar-fill {
  height: 100%;
  background: linear-gradient(90deg, var(--accent), #2d8aa0);
  border-radius: 999px;
  transition: width 0.6s ease-out;
}
.rank-1 .bar-fill {
  background: linear-gradient(90deg, var(--accent-2), #d99e2b);
}

#back-btn {
  background: transparent !important;
  color: var(--text-2) !important;
  border: 1px solid rgba(78,197,216,0.35) !important;
  font-weight: 600 !important;
  border-radius: 12px !important;
  padding: 10px 22px !important;
}
#back-btn:hover {
  background: rgba(78,197,216,0.08) !important;
  color: var(--text-1) !important;
}

.spec-block {
  margin-top: 20px;
  background: rgba(10,26,46,0.6);
  border: 1px solid rgba(78,197,216,0.15);
  border-radius: 14px;
  padding: 14px;
}
.spec-title {
  color: var(--text-2); font-size: 0.78rem;
  letter-spacing: 0.18em; text-transform: uppercase;
  margin-bottom: 10px;
}

.footnote {
  text-align: center; color: var(--text-2); opacity: 0.6;
  font-size: 0.78rem; margin-top: 28px;
}
"""

with gr.Blocks(css=CUSTOM_CSS, title="🐋 고래 종 분류기",
               theme=gr.themes.Base(primary_hue="cyan", neutral_hue="slate")) as demo:

    # 공통 헤더
    gr.HTML("""
    <div class="screen-wrap" style="padding-bottom:0">
      <div class="brand">
        <h1 class="brand-title">🐋  고래 종 분류기</h1>
        <p class="brand-sub">Whale Species Classifier · CNN-based acoustic identification</p>
      </div>
    </div>
    """)

    # ───────── 화면 1: 업로드 ─────────
    with gr.Column(visible=True, elem_classes="screen-wrap") as upload_screen:
        gr.HTML('<div class="upload-card">'
                '<div class="upload-title">🎧 고래 소리를 업로드하세요</div>'
                '<div class="upload-desc">파일을 끌어다 놓거나 마이크로 녹음하세요 · '
                'WAV, MP3, FLAC 등 지원 · 1~5초 길이의 휘슬·콜·클릭음 권장</div>'
                '</div>')
        audio_in = gr.Audio(
            sources=["upload", "microphone"],
            type="filepath",
            label="",
            show_label=False,
        )
        with gr.Row():
            gr.HTML("<div style='flex:1'></div>")
            analyze_btn = gr.Button("🔍  분류 시작하기", elem_id="analyze-btn", scale=0)
            gr.HTML("<div style='flex:1'></div>")

    # ───────── 화면 2: 결과 ─────────
    with gr.Column(visible=False, elem_classes="screen-wrap") as result_screen:
        hero_box = gr.HTML()

        gr.HTML('<div style="color:var(--text-2);font-size:0.85rem;letter-spacing:0.18em;'
                'text-transform:uppercase;margin:18px 0 10px">🏆 유사도 순위 Top 5</div>'
                '<div class="rank-list">')
        rank1_html = gr.HTML()
        rank2_html = gr.HTML()
        rank3_html = gr.HTML()
        rank4_html = gr.HTML()
        rank5_html = gr.HTML()
        gr.HTML("</div>")

        gr.HTML('<div class="spec-block">'
                '<div class="spec-title">🎼 입력 오디오의 MFCC 특징</div>')
        spec_img_out = gr.Image(show_label=False, container=False, height=240)
        gr.HTML("</div>")

        with gr.Row():
            gr.HTML("<div style='flex:1'></div>")
            back_btn = gr.Button("↺  다른 소리 다시 분석하기", elem_id="back-btn", scale=0)
            gr.HTML("<div style='flex:1'></div>")

    # 공통 푸터
    gr.HTML("""
    <div class="footnote screen-wrap">
      Reference · Suarez D. (2023) <i>Whale Detection through Sound Analysis with CNN</i>, Kaggle ·
      윤영글 외 (2024) <i>CNN 모델을 이용한 수중음향 자료의 돌고래 휘슬음 자동 분류 연구</i>, 한국수산과학회지.
    </div>
    """)

    # 이벤트 연결
    analyze_btn.click(
        go_analyze,
        inputs=[audio_in],
        outputs=[
            upload_screen, result_screen,
            hero_box,
            rank1_html, rank2_html, rank3_html, rank4_html, rank5_html,
            spec_img_out,
        ],
    )
    back_btn.click(
        go_back,
        inputs=None,
        outputs=[upload_screen, result_screen],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=False)
