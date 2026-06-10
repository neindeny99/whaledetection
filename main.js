/* ───────────────────────────────────────────────
   🐋 고래 종 분류기 — 프런트엔드 로직
   ─────────────────────────────────────────────── */

// API 주소: 같은 서버(FastAPI)가 이 페이지를 서빙하므로 상대경로면 충분.
// 프런트를 다른 도메인에서 호스팅한다면 아래를 전체 주소로 바꾸세요.
//   예: const API_BASE = "https://whale.내도메인.com";
const API_BASE = "";

const MEDALS = ["🥇", "🥈", "🥉", "④", "⑤"];

const els = {
  uploadScreen: document.getElementById("upload-screen"),
  resultScreen: document.getElementById("result-screen"),
  dropzone:     document.getElementById("dropzone"),
  dzText:       document.getElementById("dz-text"),
  fileInput:    document.getElementById("audio-input"),
  preview:      document.getElementById("preview"),
  analyzeBtn:   document.getElementById("analyze-btn"),
  backBtn:      document.getElementById("back-btn"),
  heroBox:      document.getElementById("hero-box"),
  rankHeader:   document.getElementById("rank-header"),
  rankList:     document.getElementById("rank-list"),
};

let selectedFile = null;

// ── 파일 선택 처리 ──────────────────────────────
function setFile(file) {
  if (!file) return;
  selectedFile = file;
  els.dzText.textContent = file.name;
  els.dropzone.classList.add("has-file");

  const url = URL.createObjectURL(file);
  els.preview.src = url;
  els.preview.hidden = false;

  els.analyzeBtn.disabled = false;
}

els.fileInput.addEventListener("change", (e) => setFile(e.target.files[0]));

// 키보드 접근성: 드롭존에 포커스 후 Enter/Space → 파일 선택창
els.dropzone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") {
    e.preventDefault();
    els.fileInput.click();
  }
});

// 드래그 앤 드롭
["dragenter", "dragover"].forEach((evt) =>
  els.dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    els.dropzone.classList.add("dragover");
  })
);
["dragleave", "drop"].forEach((evt) =>
  els.dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    els.dropzone.classList.remove("dragover");
  })
);
els.dropzone.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files[0];
  if (file) {
    els.fileInput.files = e.dataTransfer.files; // input 에도 반영
    setFile(file);
  }
});

// ── 분석 요청 ───────────────────────────────────
els.analyzeBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  els.analyzeBtn.classList.add("loading");
  els.analyzeBtn.disabled = true;
  els.analyzeBtn.textContent = "🌊 분석 중...";

  try {
    const form = new FormData();
    form.append("file", selectedFile);

    const res = await fetch(`${API_BASE}/predict`, { method: "POST", body: form });
    if (!res.ok) {
      let msg = `서버 오류 (${res.status})`;
      try {
        const err = await res.json();
        if (err.detail) msg = err.detail;
      } catch (_) {}
      throw new Error(msg);
    }

    const data = await res.json();
    renderResult(data);
    showScreen("result");
  } catch (err) {
    alert(`분석에 실패했습니다.\n${err.message}`);
  } finally {
    els.analyzeBtn.classList.remove("loading");
    els.analyzeBtn.disabled = false;
    els.analyzeBtn.innerHTML = "🔍&nbsp; 분류 시작하기";
  }
});

// ── 결과 렌더링 ─────────────────────────────────
function renderResult(data) {
  // 어떤 종과도 0.5 이상 일치하지 않음 → 고래 소리가 아님
  if (!data.is_whale) {
    els.heroBox.innerHTML = `
      <div class="hero-result not-whale">
        <div class="hero-label">분석 결과</div>
        <div class="hero-name">🚫 고래 소리가 아님</div>
        <div class="hero-name-ko">어떤 고래 종과도 50% 이상 일치하지 않습니다</div>
        <div class="hero-pct-sm">최고 유사도 ${(data.top_prob * 100).toFixed(2)}%</div>
      </div>`;
    els.rankHeader.innerHTML = "";
    els.rankList.innerHTML = "";
    return;
  }

  const top = data.results[0];
  els.heroBox.innerHTML = `
    <div class="hero-result">
      <div class="hero-label">가장 닮은 고래</div>
      <div class="hero-name">🐋 ${top.en}</div>
      <div class="hero-name-ko">${top.ko}</div>
      <div class="hero-pct">${(top.prob * 100).toFixed(2)}%</div>
    </div>`;

  els.rankHeader.innerHTML = '<div class="rank-header">🏆 유사도 순위 Top 5</div>';

  els.rankList.innerHTML = data.results
    .map((r, i) => {
      const pct = r.prob * 100;
      const bar = Math.max(2, pct); // 너무 작으면 안 보여서 최소 2%
      return `
        <div class="rank-card rank-${i + 1}">
          <div class="rank-medal">${MEDALS[i]}</div>
          <div class="rank-body">
            <div class="rank-line">
              <span class="rank-num">${i + 1}위</span>
              <span class="rank-name">${r.en}</span>
              <span class="rank-pct">${pct.toFixed(2)}%</span>
            </div>
            <div class="rank-name-ko">${r.ko}</div>
            <div class="bar-track"><div class="bar-fill" data-w="${bar.toFixed(2)}"></div></div>
          </div>
        </div>`;
    })
    .join("");

  // 바 애니메이션: 다음 프레임에 width 적용
  requestAnimationFrame(() => {
    els.rankList.querySelectorAll(".bar-fill").forEach((el) => {
      el.style.width = el.dataset.w + "%";
    });
  });
}

// ── 화면 전환 ───────────────────────────────────
function showScreen(which) {
  const result = which === "result";
  els.resultScreen.hidden = !result;
  els.uploadScreen.hidden = result;
  window.scrollTo({ top: 0, behavior: "smooth" });
}

els.backBtn.addEventListener("click", () => showScreen("upload"));
