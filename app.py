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
    page_title="라이니 - 선 따기 자동화",
    
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

def extract_contours(gray: np.ndarray, gap_fill: int = -1):
    """
    스마트 외곽선 추출.
    gap_fill = -1 (기본): 자동 판별
        - 최대 구멍/최대 외곽 비율 > 20% → 흰 몸통 객체(고양이 등) → 자동 gap_fill
        - 비율 ≤ 20% → 획/글자/일반 객체 → 내부 구멍 3% 이상만 포함
    gap_fill = 0  : 내부 채우기 없음 (글자·세밀한 객체)
    gap_fill > 0  : 수동으로 지정한 px만큼 내부 채우기 (라임·복잡한 객체)
    """
    h, w = gray.shape
    min_outer_area = h * w * 0.01

    # ① 밝은 배경 제거 + 노이즈 제거
    _, bg_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    fg = cv2.bitwise_not(bg_mask)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    # ② 외곽 + 구멍 전부 추출
    cnts, hier = cv2.findContours(fg, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hier is None or len(cnts) == 0:
        return [], fg

    outer_areas = [cv2.contourArea(cnts[i]) for i in range(len(cnts)) if hier[0][i][3] == -1]
    hole_areas  = [cv2.contourArea(cnts[i]) for i in range(len(cnts)) if hier[0][i][3] != -1]
    max_outer = max(outer_areas) if outer_areas else 1
    max_hole  = max(hole_areas)  if hole_areas  else 0

    # ③ gap_fill 값 결정
    if gap_fill == -1:
        # 자동 판별
        largest_hole_ratio = max_hole / max_outer
        if largest_hole_ratio > 0.20:
            outer_cnts = [cnts[i] for i in range(len(cnts))
                          if hier[0][i][3] == -1 and cv2.contourArea(cnts[i]) >= min_outer_area]
            largest = max(outer_cnts, key=cv2.contourArea) if outer_cnts else max(cnts, key=cv2.contourArea)
            _, _, bw, bh = cv2.boundingRect(largest)
            gap_fill = max(10, min(bw, bh) // 8)
        else:
            gap_fill = 0

    # ④ gap_fill 적용
    if gap_fill > 0:
        k = np.ones((gap_fill, gap_fill), np.uint8)
        filled = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k)
        binary = cv2.morphologyEx(filled, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        # RETR_EXTERNAL: 내부 구멍 완전 무시 → 고양이 몸통 내부 선 방지
        result_cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        result = [c for c in result_cnts if cv2.contourArea(c) >= min_outer_area]
        # gap_fill 적용 시 내부 칼선 방지를 위해 binary를 채워진 상태로 반환
        return result, binary
    else:
        # gap_fill=0: 외곽 전부 + 구멍은 최대 외곽의 3% 이상만
        HOLE_MIN_RATIO = 0.03
        result = []
        for i, cnt in enumerate(cnts):
            area = cv2.contourArea(cnt)
            is_hole = hier[0][i][3] != -1
            if is_hole:
                if area / max_outer >= HOLE_MIN_RATIO:
                    result.append(cnt)
            else:
                if area >= h * w * 0.001:
                    result.append(cnt)

    if not result and cnts:
        result = [max(cnts, key=cv2.contourArea)]
    return result, fg

def dilate_contours(binary: np.ndarray, offset_mm: float, gap_fill: int = -1):
    """
    칼선 추출 (실선) — 외곽선의 내부 채우기 설정과 동일한 기준 적용:
    gap_fill > 0 : binary가 이미 채워진 상태 → 외곽 칼선만, 내부 칼선 없음
    gap_fill = 0 : 채우기 없음 → 구멍(ㅁ) 3% 이상인 것에만 내부 칼선 생성
    gap_fill = -1: 자동 판별 → 구멍 비율 20% 초과 시 내부 칼선 없음
    """
    h, w = binary.shape
    offset_px = mm_to_px(offset_mm)
    k_full = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (offset_px * 2 + 1, offset_px * 2 + 1))

    # ① 외곽 칼선: 항상 생성
    dilated = cv2.dilate(binary, k_full, iterations=1)
    outer_cnts, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    result = [c for c in outer_cnts if cv2.contourArea(c) > 50]

    # ② gap_fill > 0이면 내부 칼선 없음
    if gap_fill > 0:
        return result

    # ③ 구멍 분석
    all_cnts, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None:
        return result

    outer_areas = [cv2.contourArea(all_cnts[i]) for i in range(len(all_cnts)) if hierarchy[0][i][3] == -1]
    hole_areas  = [cv2.contourArea(all_cnts[i]) for i in range(len(all_cnts)) if hierarchy[0][i][3] != -1]
    max_outer = max(outer_areas) if outer_areas else 1
    max_hole  = max(hole_areas)  if hole_areas  else 0

    # gap_fill=-1(자동): 구멍 비율 20% 초과면 흰 몸통 객체 → 내부 칼선 없음
    if gap_fill == -1 and (max_hole / max_outer) > 0.20:
        return result

    # ④ 내부 칼선: 구멍 면적이 최대 외곽의 3% 이상인 것만
    HOLE_MIN_RATIO = 0.03
    inner_cut_cnts = []
    for i, cnt in enumerate(all_cnts):
        if hierarchy[0][i][3] == -1:
            continue
        area = cv2.contourArea(cnt)
        if area / max_outer < HOLE_MIN_RATIO:
            continue
        hole_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(hole_mask, [cnt], -1, 255, -1)
        _, _, bw, bh = cv2.boundingRect(cnt)
        max_erode = max(1, min(bw, bh) // 3)
        actual_px = min(offset_px, max_erode)
        k_fit = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (actual_px * 2 + 1, actual_px * 2 + 1))
        eroded = cv2.erode(hole_mask, k_fit, iterations=1)
        found, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        valid = [c for c in found if cv2.contourArea(c) > 10]
        if valid:
            inner_cut_cnts.extend(valid)
        else:
            inner_cut_cnts.append(cnt)

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
                 smoothing=2.0, gap_fill=-1):
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
        cut_contours = dilate_contours(binary, cutline_offset_mm, gap_fill=gap_fill)
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
                         smoothing=2.0, gap_fill=-1):
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
        cut_contours = dilate_contours(binary, cutline_offset_mm, gap_fill=gap_fill)
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

LOGO_B64 = "/9j/4AAQSkZJRgABAQAAAQABAAD/4gHYSUNDX1BST0ZJTEUAAQEAAAHIAAAAAAQwAABtbnRyUkdCIFhZWiAH4AABAAEAAAAAAABhY3NwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAA9tYAAQAAAADTLQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAlkZXNjAAAA8AAAACRyWFlaAAABFAAAABRnWFlaAAABKAAAABRiWFlaAAABPAAAABR3dHB0AAABUAAAABRyVFJDAAABZAAAAChnVFJDAAABZAAAAChiVFJDAAABZAAAAChjcHJ0AAABjAAAADxtbHVjAAAAAAAAAAEAAAAMZW5VUwAAAAgAAAAcAHMAUgBHAEJYWVogAAAAAAAAb6IAADj1AAADkFhZWiAAAAAAAABimQAAt4UAABjaWFlaIAAAAAAAACSgAAAPhAAAts9YWVogAAAAAAAA9tYAAQAAAADTLXBhcmEAAAAAAAQAAAACZmYAAPKnAAANWQAAE9AAAApbAAAAAAAAAABtbHVjAAAAAAAAAAEAAAAMZW5VUwAAACAAAAAcAEcAbwBvAGcAbABlACAASQBuAGMALgAgADIAMAAxADb/2wBDAAUDBAQEAwUEBAQFBQUGBwwIBwcHBw8LCwkMEQ8SEhEPERETFhwXExQaFRERGCEYGh0dHx8fExciJCIeJBweHx7/2wBDAQUFBQcGBw4ICA4eFBEUHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh7/wAARCAMABYADASIAAhEBAxEB/8QAHAABAQADAAMBAAAAAAAAAAAAAAEGBwgCBAUD/8QARxABAAEDAwEFBQQHBQcDBAMAAAECAwQFBhEhBxIxQVEIImFxgRMUkaEVIzJCUrHBJFNigtElM0NykqKyY9LwFkSj4WRzwv/EABoBAQADAQEBAAAAAAAAAAAAAAADBAUBAgb/xAAqEQEAAgIBBAEDBAMBAQAAAAAAAQIDBBEFEiExEyJBURQjMmEzQoEVNP/aAAwDAQACEQMRAD8A7JjwWSFBBQDyTrz4KAIoAigIKAIoCEKAJ1UATqoCCgIKAIoCLAAIoAnVQBFAEUARQBFAE6qAJ1UARQBOqgCKAJ1UAhFAEUATqoBCdVAE6qAQigCKAIoAnVQBFAE6qAQnVQBOqgEJ1UATqoAnVQBOqgCdfRQB49fR5AJB19FAITryoAnX0UAhOqgCdfRQCE6qAHXnwABOvooBCfRQCE6qAnU6qAJ19FAIRQCEUATqoAR4gAnVQBFAEUARQBFAEUARQBOqgCdVAEUARQBOqgCdVAE6qAIoAigB1ABOqgCdfRQBFAEUATqoAigCKAnJzKgCKAIoCR4qkeKgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAkeKkAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAJHiqeagAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAkKQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQJCgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAkKlKgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA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AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAJCpCgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACQoAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAECQoAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACQqQoAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAJCpCgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACQoAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAJCpCgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAkKkKAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACKkKAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACKkKAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACKkKAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACQqQoAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAkKkKAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABAkKAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACQqQoAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAhB09T6gohz8YBROfivPyADmPWDmAA5j1OY9QA5j1OY9QA5j1AA5j1AA5j1OYADmPU5gAOQAOY9TmAA5j1OY9QA5j1OYADmPU5gAOY9TmAA5j1OYADmPU5gAOY9TmAA5j1OY9QA5j1OYADmPUADmPUADmPUADkADmPU5gAOYOYADmPUADmPUADmPUADmAAOTmAA5AA5j1OYADkADmPUADmPUADmPUADk5gAOY9QAOY9TmAA5OYADmAAOYOYADk5gAOQAOY9TmAA5j1AA5j1AA5j1OYADmPU5j1ADmPU5j1ADmPU5gAOY9TmAA5j1OYADmPU5gAOY9QAOY9QAOY9QAOY9QAOQAOQAOY9TmAA5AA5j1OYADmDmAA5j1AA5gADmPUADmPUADmPUADmPU5j1ADk5gAOY9TmAA5j1AA5j1AA5j1AA5j1OYADmPUADmPU5gAOY9TmAA5j1OYADmPU5gAOY9TmPUAOY9QAOY9QAOeoAHJM9Af/9k="

st.markdown(f"""
<div style="display:flex; align-items:center; gap:14px; margin-bottom:2px;">
  <img src="data:image/png;base64,{LOGO_B64}"
       style="height:64px; width:auto; object-fit:contain;">
  <div>
    <div style="font-size:2rem; font-weight:800; color:#1a6b7c; line-height:1.1;">
      라이니
    </div>
    <div style="font-size:0.95rem; color:#555; margin-top:2px;">
      복잡한 선 따기는 끝, 클릭 한 번으로 완성하세요
    </div>
  </div>
</div>
""", unsafe_allow_html=True)
st.divider()

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
        cutline_offset_mm = st.number_input("칼선 간격 (mm)", 0.1, 30.0, 2.0, 0.5)
        cutline_width_mm  = st.number_input("칼선 두께 (mm)", 0.1, 5.0, 0.3, 0.1)
        cutline_color     = st.color_picker("칼선 색상", "#FF0000")
    else:
        cutline_offset_mm, cutline_width_mm, cutline_color = 2.0, 0.3, "#FF0000"

if not use_outline and not use_cutline:
    st.warning("외곽선 또는 칼선을 하나는 활성화해주세요.")

st.markdown("**선 스무딩**")
st.markdown("""
<table style="width:100%; font-size:0.8rem; border-collapse:collapse; margin-bottom:6px;">
  <tr>
    <td style="padding:8px 12px; width:33%; text-align:center; background:#f0f2f6; border-radius:6px 0 0 6px; border:1px solid #ddd;">
      <b>약하게</b><br>
      <span style="color:#555;">원본 형태 그대로 유지<br>픽셀 노이즈만 제거<br>글자·세밀한 객체 추천</span>
    </td>
    <td style="padding:8px 12px; width:33%; text-align:center; background:#e8f4fd; border:1px solid #90caf9;">
      <b>보통 (기본)</b><br>
      <span style="color:#555;">자연스럽게 부드러운 선<br>대부분의 이미지에 적합<br>기본 권장값</span>
    </td>
    <td style="padding:8px 12px; width:33%; text-align:center; background:#fff3cd; border-radius:0 6px 6px 0; border:1px solid #ffc107;">
      <b>많이</b><br>
      <span style="color:#555;">모서리가 둥글게 처리됨<br>단순한 실루엣에 적합<br>글자는 형태 손실 주의</span>
    </td>
  </tr>
</table>
""", unsafe_allow_html=True)
smoothing = st.select_slider(
    "스무딩 강도",
    options=["약하게", "보통", "많이"],
    value="보통"
)
smoothing_val = {"약하게": 1, "보통": 4, "많이": 8}[smoothing]

st.markdown("**내부 채우기**")
# 자동 체크박스
gap_fill_auto = st.checkbox("내부 채우기 자동 판별", value=True,
    help="이미지 유형을 분석해 고양이·스티커는 채우기 적용, 글자·단색 객체는 채우기 없음으로 자동 결정합니다.")

if gap_fill_auto:
    # 자동 선택 시 안내만 표시
    st.markdown("""
<div style="padding:8px 14px; background:#e8f4fd; border-left:4px solid #0078D4;
            border-radius:4px; font-size:0.85rem; color:#333; margin-bottom:8px;">
  이미지 유형 자동 판별 중 — 고양이·스티커는 채우기 적용 /
  글자·한자·단색 객체는 채우기 없음
</div>
""", unsafe_allow_html=True)
    gap_fill = -1
else:
    st.markdown("""
<table style="width:100%; font-size:0.8rem; border-collapse:collapse; margin-bottom:6px;">
  <tr>
    <td style="padding:8px 12px; width:33%; text-align:center; background:#f0f2f6; border-radius:6px 0 0 6px; border:1px solid #ddd;">
      <b>끔</b><br>
      <span style="color:#555;">내부 채우기 사용 안 함<br>글자·한자처럼 획이 객체이고<br>내부 구멍(ㅁ)에도 선이 필요할 때</span>
    </td>
    <td style="padding:8px 12px; width:33%; text-align:center; background:#e8f4fd; border:1px solid #90caf9;">
      <b>중간 (기본)</b><br>
      <span style="color:#555;">내부 디테일 일부 메움<br>자동 판별이 애매한 경우<br>라임 등 중간 복잡도 이미지</span>
    </td>
    <td style="padding:8px 12px; width:33%; text-align:center; background:#fff3cd; border-radius:0 6px 6px 0; border:1px solid #ffc107;">
      <b>최대</b><br>
      <span style="color:#555;">내부 디테일 모두 메워 외곽만<br>병·라임처럼 내부에 선이<br>불필요하게 생기는 경우</span>
    </td>
  </tr>
</table>
""", unsafe_allow_html=True)
    gap_fill_sel = st.select_slider(
        "내부 채우기 강도",
        options=["끔", "중간", "최대"],
        value="중간"
    )
    gap_fill = {"끔": 0, "중간": 15, "최대": 30}[gap_fill_sel]

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
                        smoothing=smoothing_val, gap_fill=gap_fill
                    )
                    bar.progress(65, text="미리보기 생성 중...")

                    # 미리보기 이미지 생성
                    preview_img = generate_preview_img(
                        img_bgr,
                        use_outline, outline_mm, outline_color,
                        use_cutline, cutline_offset_mm, cutline_width_mm, cutline_color,
                        smoothing=smoothing_val, gap_fill=gap_fill
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
