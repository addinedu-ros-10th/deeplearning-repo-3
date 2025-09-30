from ultralytics import YOLO
import cv2
import numpy as np
from collections import deque
from api import route_stream

# -----------------------
# 모델 로드
# -----------------------
model_paths = {
    "Dynamic": "Obstacle_detect.pt",
    "Static": "Obstacle_detect.pt",
    "Surface": "Surface_detect.pt"
}
models = {name: YOLO(path) for name, path in model_paths.items()}

# -----------------------
# 허용 클래스 정의
# -----------------------
allowed_dynamic = {"person", "bicycle", "motorcycle", "scooter", "car"}
static_ground_allowed = {"tree_trunk", "fire_hydrant", "stop",
                         "pole", "bollard", "barricade", "bench",
                         "movable_signage", "parking_meter"}
static_head_allowed = {"traffic_light"}
surface_ground_allowed = {"caution_zone"}

# -----------------------
# ROI 함수
# -----------------------
def get_ground_roi(W, H):
    bottom_width = 0.7
    top_width = 0.3
    cx = W // 2
    cy_bottom = H
    cy_top = int(H * 0.7)

    bl = (int(cx - W * bottom_width / 2), cy_bottom)
    br = (int(cx + W * bottom_width / 2), cy_bottom)
    tr = (int(cx + W * top_width / 2), cy_top)
    tl = (int(cx - W * top_width / 2), cy_top)

    roi = np.array([[bl, br, tr, tl]], dtype=np.int32)
    left_poly = np.array([[bl, (cx, cy_bottom), (cx, cy_top), tl]], dtype=np.int32)
    right_poly = np.array([[(cx, cy_bottom), br, tr, (cx, cy_top)]], dtype=np.int32)

    return roi, left_poly, right_poly, cy_top

def get_head_roi(W, H, top_width_ratio=0.8, bottom_width_ratio=0.4):
    top_y = 0
    bottom_y = int(H * 0.25)
    cx = W // 2
    top_half = int(W * top_width_ratio / 2)
    bottom_half = int(W * bottom_width_ratio / 2)

    roi_points = np.array([[
        (cx - top_half, top_y),
        (cx + top_half, top_y),
        (cx + bottom_half, bottom_y),
        (cx - bottom_half, bottom_y)
    ]], dtype=np.int32)

    return roi_points

# -----------------------
# 궤적 추적 버퍼
# -----------------------
trajectory_buffer = {}
max_history = 5

def update_trajectory(obj_id, cx, cy):
    from collections import deque
    if obj_id not in trajectory_buffer:
        trajectory_buffer[obj_id] = deque(maxlen=max_history)
    trajectory_buffer[obj_id].append((cx, cy))
    if len(trajectory_buffer[obj_id]) >= 2:
        (x1, y1), (x2, y2) = trajectory_buffer[obj_id][-2], trajectory_buffer[obj_id][-1]
        return x2 - x1, y2 - y1
    return 0, 0

# -----------------------
# YOLO 처리 함수
# -----------------------
async def detect(color_file, depth_file):
    # 이미지 복원
    color_bytes = await color_file.read()
    color_arr = np.frombuffer(color_bytes, np.uint8)
    color_img = cv2.imdecode(color_arr, cv2.IMREAD_COLOR)

    depth_bytes = await depth_file.read()
    depth_arr = np.frombuffer(depth_bytes, np.uint8)
    depth_img = cv2.imdecode(depth_arr, cv2.IMREAD_UNCHANGED)  # uint16 깊이

    H, W = color_img.shape[:2]

    # ROI 계산
    ground_roi, ground_left, ground_right, horizon_y = get_ground_roi(W, H)
    head_roi = get_head_roi(W, H)

    ground_left_state = "safe"
    ground_right_state = "safe"
    head_state = "safe"

    detections = []

    # 여러 모델 순차 적용
    for model_name, model in models.items():
        results = model(color_img, conf=0.5)
        names = model.names

        for i, box in enumerate(results[0].boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            label = names[cls]

            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            ref_y = int(y2 - (y2 - y1) * 0.125)

            if ref_y < 0 or ref_y >= depth_img.shape[0] or cx < 0 or cx >= depth_img.shape[1]:
                continue

            depth_val = depth_img[ref_y, cx]
            depth_m = float(depth_val) / 1000.0 if depth_val > 0 else -1.0

            detections.append({
                "model": model_name,
                "class": label,
                "depth_m": depth_m
            })

            # -------------------
            # Dynamic 처리
            # -------------------
            if model_name == "Dynamic":
                if label not in allowed_dynamic:
                    continue
                if cv2.pointPolygonTest(ground_left[0], (cx, ref_y), False) >= 0:
                    if 0 < depth_m <= 2.0:
                        ground_left_state = "warning"
                    elif 2.0 < depth_m <= 3.0:
                        ground_left_state = "caution"
                elif cv2.pointPolygonTest(ground_right[0], (cx, ref_y), False) >= 0:
                    if 0 < depth_m <= 2.0:
                        ground_right_state = "warning"
                    elif 2.0 < depth_m <= 3.0:
                        ground_right_state = "caution"

                update_trajectory(f"{model_name}_{i}", cx, ref_y)

            # -------------------
            # Static 처리
            # -------------------
            elif model_name == "Static":
                # 상단
                if cv2.pointPolygonTest(head_roi[0], (cx, y1), False) >= 0:
                    if label in static_head_allowed:
                        print(label, static_head_allowed)
                        if 0 < depth_m <= 2.0:
                            head_state = "warning"
                        elif 2.0 < depth_m <= 3.0:
                            head_state = "caution"
                # 하단 좌/우
                if cv2.pointPolygonTest(ground_left[0], (cx, ref_y), False) >= 0:
                    if label in static_ground_allowed:
                        if 0 < depth_m <= 2.0:
                            ground_left_state = "warning"
                        elif 2.0 < depth_m <= 3.0:
                            ground_left_state = "caution"
                if cv2.pointPolygonTest(ground_right[0], (cx, ref_y), False) >= 0:
                    if label in static_ground_allowed:
                        if 0 < depth_m <= 2.0:
                            ground_right_state = "warning"
                        elif 2.0 < depth_m <= 3.0:
                            ground_right_state = "caution"

            # -------------------
            # Surface 처리
            # -------------------
            elif model_name == "Surface":
                if label not in surface_ground_allowed:
                    continue
                if cv2.pointPolygonTest(ground_left[0], (cx, ref_y), False) >= 0:
                    if 0 < depth_m <= 2.0:
                        ground_left_state = "warning"
                    elif 2.0 < depth_m <= 3.0:
                        ground_left_state = "caution"
                elif cv2.pointPolygonTest(ground_right[0], (cx, ref_y), False) >= 0:
                    if 0 < depth_m <= 2.0:
                        ground_right_state = "warning"
                    elif 2.0 < depth_m <= 3.0:
                        ground_right_state = "caution"

    # 최종 threat_level
    if "warning" in [ground_left_state, ground_right_state, head_state]:
        threat_level = "high"
    elif "caution" in [ground_left_state, ground_right_state, head_state]:
        threat_level = "medium"
    else:
        threat_level = "low"
    
    # ROI 시각화
    overlay = color_img.copy()
    def state_to_color(state):
        if state == "safe":
            return (0, 255, 0)
        if state == "caution":
            return (0, 255, 255)
        if state == "warning":
            return (0, 0, 255)
        return (50, 50, 50)

    left_color = state_to_color(ground_left_state)
    right_color = state_to_color(ground_right_state)
    head_color = state_to_color(head_state)

    cv2.fillPoly(overlay, ground_left, left_color)
    cv2.fillPoly(overlay, ground_right, right_color)
    cv2.fillPoly(overlay, head_roi, head_color)

    display = cv2.addWeighted(overlay, 0.35, color_img, 0.65, 0)
    cv2.polylines(display, ground_left, True, left_color, 2)
    cv2.polylines(display, ground_right, True, right_color, 2)
    cv2.polylines(display, head_roi, True, head_color, 2)

    cv2.putText(display,
                f"Ground-L: {ground_left_state}  Ground-R: {ground_right_state}  Head: {head_state}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    route_stream.last_frame = display.copy()

    return detections, {
        "ground_left": ground_left_state,
        "ground_right": ground_right_state,
        "head": head_state
    }, threat_level
