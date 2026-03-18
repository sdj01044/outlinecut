import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
from scipy.ndimage import gaussian_filter1d

# ─────────────────────────────────────────
#  페이지 설정
# ─────────────────────────────────────────
st.set_page_config(
    page_title="EPS 외곽선 생성기",
    
    layout="wide"
)

st.markdown("""
<style>
    .stApp { background-color: #f8f9fa; }
    .block-container { padding-top: 2rem; }
    h1 { color: #1a1a2e; font-size: 1.8rem; }
    .stDownloadButton > button {
        background-color: #0078D4;
        color: white;
        border: none;
        border-radius: 8px;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
#  헬퍼 함수
# ─────────────────────────────────────────
DPI = 96  # 화면 기준 DPI

# ── 가우시안 스무딩 ──
def smooth_contour(cnt, sigma: float = 2.0):
    """
    점 위치를 가우시안 필터로 부드럽게 이동.
    점 수는 그대로 유지 → 글자/복잡한 형태도 정확도 손실 없음.
    sigma: 1=약하게, 3=보통, 6=많이 스무딩
    """
    pts = cnt.reshape(-1, 2).astype(float)
    if len(pts) < 3:
        return pts
    xs = gaussian_filter1d(pts[:, 0], sigma=sigma, mode='wrap')
    ys = gaussian_filter1d(pts[:, 1], sigma=sigma, mode='wrap')
    return np.stack([xs, ys], axis=1)

def pts_to_eps_path(pts, img_h, sx=1.0, sy=1.0):
    """스무딩된 포인트 → EPS lineto 패스"""
    n = len(pts)
    if n < 3:
        return ""
    lines = [f"{pts[0][0]*sx:.3f} {(img_h - pts[0][1])*sy:.3f} moveto"]
    for p in pts[1:]:
        lines.append(f"{p[0]*sx:.3f} {(img_h - p[1])*sy:.3f} lineto")
    lines.append("closepath")
    return "\n".join(lines)

def mm_to_px(mm: float) -> int:
    return max(1, int(round(mm * DPI / 25.4)))

def hex_to_eps_rgb(hex_color: str):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return r/255, g/255, b/255

def extract_contours(gray: np.ndarray, gap_fill: int = 0):
    """
    스마트 외곽선 추출.
    gap_fill=0  : 획 사이를 메우지 않음 → 글자·복잡한 객체에 적합
    gap_fill>0  : gap_fill px 크기만큼 내부 공간을 메움
                  → 병처럼 라벨이 있거나 배경이 복잡한 객체에 적합
    """
    h, w = gray.shape

    # ① 밝은 배경 제거
    _, bg_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    fg_mask = cv2.bitwise_not(bg_mask)

    # ② gap_fill이 있을 때만 MORPH_CLOSE 적용
    if gap_fill > 0:
        k_close = np.ones((gap_fill, gap_fill), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, k_close)

    # ③ 작은 노이즈 제거
    k_open = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, k_open)

    # ④ RETR_CCOMP: 외곽 + 구멍(ㅁ 안쪽) 모두 추출
    cnts, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None:
        return [], binary

    min_area = h * w * 0.001
    result = [c for c in cnts if cv2.contourArea(c) >= min_area]
    if not result and cnts:
        result = [max(cnts, key=cv2.contourArea)]

    return result, binary

def dilate_contours(binary: np.ndarray, offset_mm: float):
    """
    칼선 추출 (실선):
    - 외곽 칼선: 글자 바깥으로 offset mm 팽창
    - 내부 칼선: 'ㅁ' 같은 구멍 경계에서 안쪽으로 offset mm 침식
                 구멍이 작으면 구멍 크기에 맞게 침식량 자동 조절
    """
    h, w = binary.shape
    offset_px = mm_to_px(offset_mm)
    k_full = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (offset_px * 2 + 1, offset_px * 2 + 1))

    # ① 외곽 칼선: binary를 바깥으로 팽창
    dilated = cv2.dilate(binary, k_full, iterations=1)
    outer_cnts, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # ② 내부 칼선: 구멍별로 개별 처리
    all_cnts, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    inner_cut_cnts = []
    if hierarchy is not None:
        for i, cnt in enumerate(all_cnts):
            if hierarchy[0][i][3] == -1:  # parent 없음 = 외곽 → 스킵
                continue
            if cv2.contourArea(cnt) < 30:
                continue

            # 구멍 하나를 마스크에 채움
            hole_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(hole_mask, [cnt], -1, 255, -1)

            # 구멍 크기 대비 침식량 조절 (짧은 변의 1/3 이내로 제한)
            _, _, bw, bh = cv2.boundingRect(cnt)
            max_erode = max(1, min(bw, bh) // 3)
            actual_px = min(offset_px, max_erode)
            k_fit = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (actual_px * 2 + 1, actual_px * 2 + 1))

            # 안쪽으로 침식 → 구멍 경계에서 offset mm 안쪽 라인
            eroded = cv2.erode(hole_mask, k_fit, iterations=1)
            found, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            valid = [c for c in found if cv2.contourArea(c) > 10]

            if valid:
                inner_cut_cnts.extend(valid)
            else:
                # 침식 후 아무것도 없으면 원본 구멍 경계를 칼선으로 사용
                inner_cut_cnts.append(cnt)

    result = [c for c in outer_cnts if cv2.contourArea(c) > 50]
    result.extend(inner_cut_cnts)
    return result

def contours_to_eps_paths(contours, img_h, scale_x, scale_y, smoothing=2.0):
    """컨투어 → 가우시안 스무딩된 EPS 패스 리스트"""
    paths = []
    for cnt in contours:
        pts = smooth_contour(cnt, sigma=smoothing)
        if len(pts) < 3:
            continue
        path = pts_to_eps_path(pts, img_h, scale_x, scale_y)
        if path:
            paths.append(path)
    return paths

def generate_eps(img_bgr, use_outline, outline_mm, outline_color,
                 use_cutline, cutline_offset_mm, cutline_width_mm, cutline_color,
                 smoothing=2.0, gap_fill=0):
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    pts_w = w * 72 / DPI
    pts_h = h * 72 / DPI
    scale_x = pts_w / w
    scale_y = pts_h / h

    contours, binary = extract_contours(gray, gap_fill=gap_fill)

    eps_lines = [
        "%!PS-Adobe-3.0 EPSF-3.0",
        f"%%BoundingBox: 0 0 {int(pts_w)} {int(pts_h)}",
        f"%%HiResBoundingBox: 0.0 0.0 {pts_w:.4f} {pts_h:.4f}",
        "%%Title: Outline Cut Path",
        "%%Creator: EPS 외곽선 생성기",
        "%%EndComments", "",
    ]

    if use_outline and contours:
        r, g, b = hex_to_eps_rgb(outline_color)
        stroke_pt = outline_mm * 72 / 25.4
        paths = contours_to_eps_paths(contours, h, scale_x, scale_y, smoothing)
        eps_lines += [
            "% ── 외곽선 (Outline) ──",
            f"{r:.4f} {g:.4f} {b:.4f} setrgbcolor",
            f"{stroke_pt:.4f} setlinewidth",
            "1 setlinejoin", "1 setlinecap",
        ]
        for p in paths:
            eps_lines += ["newpath", p, "stroke", ""]

    if use_cutline and contours:
        r, g, b = hex_to_eps_rgb(cutline_color)
        stroke_pt = cutline_width_mm * 72 / 25.4
        cut_contours = dilate_contours(binary, cutline_offset_mm)
        if cut_contours:
            paths = contours_to_eps_paths(cut_contours, h, scale_x, scale_y, smoothing)
            eps_lines += [
                "% ── 칼선 (Cut Line) ──",
                f"{r:.4f} {g:.4f} {b:.4f} setrgbcolor",
                f"{stroke_pt:.4f} setlinewidth",
                "1 setlinejoin", "1 setlinecap",
            ]
            for p in paths:
                eps_lines += ["newpath", p, "stroke", ""]

    eps_lines += ["", "%%EOF"]
    return "\n".join(eps_lines).encode("utf-8")


def hex_to_bgr(hex_color: str):
    """#RRGGBB → OpenCV BGR 튜플"""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)

def generate_preview_img(img_bgr, use_outline, outline_mm, outline_color,
                         use_cutline, cutline_offset_mm, cutline_width_mm, cutline_color,
                         smoothing=2.0, gap_fill=0):
    """원본 이미지 위에 외곽선/칼선을 직접 그려 RGB numpy 배열로 반환"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    contours, binary = extract_contours(gray, gap_fill=gap_fill)

    # 원본 이미지를 살짝 밝게 해서 배경으로 사용
    preview = img_bgr.copy()
    # 흰 배경에 원본을 40% 투명도로 합성
    white = np.ones_like(preview) * 255
    preview = cv2.addWeighted(preview, 0.5, white, 0.5, 0)

    if use_outline and contours:
        color = hex_to_bgr(outline_color)
        thickness = max(1, mm_to_px(outline_mm))
        for cnt in contours:
            pts = smooth_contour(cnt, sigma=smoothing).astype(np.int32)
            cv2.polylines(preview, [pts], True, color, thickness, cv2.LINE_AA)

    if use_cutline and contours:
        cut_contours = dilate_contours(binary, cutline_offset_mm)
        color = hex_to_bgr(cutline_color)
        thickness = max(1, mm_to_px(cutline_width_mm))
        for cnt in cut_contours:
            pts = smooth_contour(cnt, sigma=smoothing).astype(np.int32)
            cv2.polylines(preview, [pts], True, color, thickness, cv2.LINE_AA)

    # BGR → RGB 변환 후 반환
    return cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)


# ─────────────────────────────────────────
#  UI
# ─────────────────────────────────────────
st.title("EPS 외곽선 생성기")
st.caption("이미지를 업로드하면 외곽선과 칼선을 자동 추출하여 EPS 파일로 변환합니다.")

# 파일 크기 제한 (MB)
MAX_FILE_MB = 10

# ── 설정 (제목 아래) ──
st.subheader("설정")

col_outline, col_cutline = st.columns(2)

with col_outline:
    st.markdown("**외곽선**")
    use_outline = st.checkbox("외곽선 사용", value=True)
    if use_outline:
        outline_mm    = st.number_input("외곽선 두께 (mm)", 0.1, 10.0, 0.5, 0.1)
        outline_color = st.color_picker("외곽선 색상", "#0078D4")
    else:
        outline_mm, outline_color = 0.5, "#0078D4"

with col_cutline:
    st.markdown("**칼선**")
    use_cutline = st.checkbox("칼선 사용", value=True)
    if use_cutline:
        cutline_offset_mm = st.number_input("칼선 간격 (mm)", 0.1, 30.0, 3.0, 0.5)
        cutline_width_mm  = st.number_input("칼선 두께 (mm)", 0.1, 5.0, 0.3, 0.1)
        cutline_color     = st.color_picker("칼선 색상", "#FF0000")
    else:
        cutline_offset_mm, cutline_width_mm, cutline_color = 3.0, 0.3, "#FF0000"

if not use_outline and not use_cutline:
    st.warning("외곽선 또는 칼선을 하나는 활성화해주세요.")

st.markdown("**선 스무딩**")
smoothing = st.slider(
    "스무딩 강도 — 형태 유지하며 선을 부드럽게 (1=약하게, 6=많이)",
    min_value=1, max_value=10, value=2, step=1
)

st.markdown("**내부 채우기**")
st.caption("글자/세밀한 객체: 0 | 라벨 있는 병·복잡한 배경: 10~20")
gap_fill = st.slider(
    "객체 내부 빈 공간을 메우는 강도 (0=끔, 클수록 더 많이 채움)",
    min_value=0, max_value=30, value=0, step=1
)

st.divider()

# ── 파일 업로드 ──
uploaded_files = st.file_uploader(
    "이미지 업로드 (여러 개 선택 가능)",
    type=["png", "jpg", "jpeg", "bmp", "tiff"],
    accept_multiple_files=True,
)

if uploaded_files:
    # 파일 크기 검사
    oversized = []
    valid_files = []
    for f in uploaded_files:
        f.seek(0, 2)              # 파일 끝으로 이동
        size_mb = f.tell() / (1024 * 1024)
        f.seek(0)                 # 포인터 초기화
        if size_mb > MAX_FILE_MB:
            oversized.append((f.name, size_mb))
        else:
            valid_files.append(f)

    # 초과 파일 경고
    if oversized:
        for name, size_mb in oversized:
            st.error(
                f"파일 크기 초과: {name} ({size_mb:.1f}MB) — "
                f"최대 허용 크기는 {MAX_FILE_MB}MB입니다. 해당 파일은 처리에서 제외됩니다."
            )

    if valid_files:
        st.info(f"{len(valid_files)}개 파일 준비 완료" +
                (f" ({len(oversized)}개 제외됨)" if oversized else ""))

        can_start = use_outline or use_cutline
        if st.button("처리 시작", type="primary", use_container_width=True, disabled=not can_start):
            st.divider()
            st.subheader("처리 결과")

            for idx, file in enumerate(valid_files):
                st.markdown(f"**[{idx+1}/{len(valid_files)}] {file.name}**")
                bar = st.progress(0, text="이미지 로딩 중...")

                try:
                    # 이미지 로드 (포인터 초기화 후 읽기)
                    file.seek(0)
                    raw = np.frombuffer(file.read(), np.uint8)
                    img_bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
                    if img_bgr is None:
                        st.error(f"{file.name}: 이미지를 읽을 수 없습니다.")
                        bar.empty()
                        continue

                    bar.progress(25, text="벡터 패스 추출 중...")

                    # EPS 생성
                    eps_bytes = generate_eps(
                        img_bgr,
                        use_outline, outline_mm, outline_color,
                        use_cutline, cutline_offset_mm, cutline_width_mm, cutline_color,
                        smoothing=smoothing, gap_fill=gap_fill
                    )
                    bar.progress(65, text="미리보기 생성 중...")

                    # 미리보기 이미지 생성
                    preview_img = generate_preview_img(
                        img_bgr,
                        use_outline, outline_mm, outline_color,
                        use_cutline, cutline_offset_mm, cutline_width_mm, cutline_color,
                        smoothing=smoothing, gap_fill=gap_fill
                    )
                    bar.progress(95, text="마무리 중...")

                    # 결과 출력
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        tab1, tab2 = st.tabs(["원본", "외곽선 미리보기"])
                        with tab1:
                            st.image(img_bgr[:, :, ::-1], use_container_width=True)
                        with tab2:
                            st.image(preview_img, use_container_width=True)

                    with col2:
                        h_px, w_px = img_bgr.shape[:2]
                        st.metric("가로", f"{w_px*25.4/DPI:.1f} mm")
                        st.metric("세로", f"{h_px*25.4/DPI:.1f} mm")
                        st.metric("EPS 크기", f"{len(eps_bytes)/1024:.1f} KB")
                        eps_filename = file.name.rsplit(".", 1)[0] + ".eps"
                        st.download_button(
                            label="EPS 다운로드",
                            data=eps_bytes,
                            file_name=eps_filename,
                            mime="application/postscript",
                            key=f"dl_{idx}",
                            use_container_width=True,
                        )

                    bar.progress(100, text="완료!")

                except Exception as e:
                    st.error(f"오류 발생 ({file.name}): {str(e)}")
                    bar.empty()

                st.divider()
