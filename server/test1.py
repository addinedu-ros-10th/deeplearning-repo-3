import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
from collections import deque

# ------------------------------
# 1. YOLO 모델 불러오기 (사용자 지정)
# ------------------------------
model_paths = {
    "Dynamic": "Obstacle_detect.pt",
    "Static": "Obstacle_detect.pt",
    "Surface": "Surface_detect.pt"   # ✅ 노면 인식 모델 추가
}

models = {
    name: YOLO(path) for name, path in model_paths.items()
}

# ------------------------------
# 2. 모델별/영역별 사용할 클래스 선택 (사용자가 수정)
# ------------------------------
allowed_dynamic = {"person","bicycle","motorcycle","scooter","car"}            # 동적 모델에서 허용할 클래스
static_ground_allowed = {"tree_trunk","fire_hydrant","stop",
                         "pole","bollard","barricade","bench","movable_signage","parking_meter"}       # 하단(ground)에서 인식할 정적 클래스
static_head_allowed = {"traffic_light"}  # 상단(head)에서 인식할 정적 클래스
surface_ground_allowed = {"caution_zone"}  # ✅ 노면 장애물 클래스 지정

# ------------------------------
# 3. ROI 정의 함수
# ------------------------------
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
    left_poly = np.array([[
        bl,
        (cx, cy_bottom),
        (cx, cy_top),
        tl
    ]], dtype=np.int32)
    right_poly = np.array([[
        (cx, cy_bottom),
        br,
        tr,
        (cx, cy_top)
    ]], dtype=np.int32)
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

# ------------------------------
# 4. 동적 객체 궤적 추적용 버퍼
# ------------------------------
trajectory_buffer = {}
max_history = 5

def update_trajectory(obj_id, cx, cy):
    if obj_id not in trajectory_buffer:
        trajectory_buffer[obj_id] = deque(maxlen=max_history)
    trajectory_buffer[obj_id].append((cx, cy))
    if len(trajectory_buffer[obj_id]) >= 2:
        (x1, y1), (x2, y2) = trajectory_buffer[obj_id][-2], trajectory_buffer[obj_id][-1]
        dx, dy = x2 - x1, y2 - y1
        return dx, dy
    return 0, 0

# ------------------------------
# 5. RealSense 파이프라인 + 메인 루프
# ------------------------------
pipeline = None
try:
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)

    align_to = rs.stream.color
    align = rs.align(align_to)
    pipeline.start(config)

    while True:
        frames = pipeline.wait_for_frames(timeout_ms=10000)
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        H, W = color_image.shape[:2]

        ground_roi, ground_left, ground_right, horizon_y = get_ground_roi(W, H)
        head_roi = get_head_roi(W, H)

        ground_left_state = "safe"
        ground_right_state = "safe"
        head_state = "safe"

        # ------------------------------
        # 모델별 순차 적용
        # ------------------------------
        for model_name, model in models.items():
            results = model(color_image, conf=0.5)
            names = model.names

            for i, box in enumerate(results[0].boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                label = names[cls]

                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                ref_y = int(y2 - (y2 - y1) * 0.125)
                depth = depth_frame.get_distance(cx, ref_y)

                # ------------------------------
                # Dynamic 처리
                # ------------------------------
                if model_name == "Dynamic":
                    if label not in allowed_dynamic:
                        continue
                    if cv2.pointPolygonTest(ground_left[0], (cx, ref_y), False) >= 0:
                        if depth <= 2.0:
                            ground_left_state = "warning"
                        elif 2.0 < depth <= 3.0:
                            ground_left_state = "caution"
                    elif cv2.pointPolygonTest(ground_right[0], (cx, ref_y), False) >= 0:
                        if depth <= 2.0:
                            ground_right_state = "warning"
                        elif 2.0 < depth <= 3.0:
                            ground_right_state = "caution"

                    dx, dy = update_trajectory(f"{model_name}_{i}", cx, ref_y)
                    cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 200, 0), 2)
                    cv2.putText(color_image, f"{model_name}:{label}:{depth:.2f}m",
                                (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 200, 0), 1)

                # ------------------------------
                # Static 처리
                # ------------------------------
                elif model_name == "Static":
                    # 상단 ROI
                    if cv2.pointPolygonTest(head_roi[0], (cx, y1), False) >= 0:
                        if label in static_head_allowed:
                            if depth <= 2.0:
                                head_state = "warning"
                            elif 2.0 < depth <= 3.0:
                                head_state = "caution"
                            else:
                                head_state = "safe"
                            cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 0, 180), 2)
                            cv2.putText(color_image, f"STATIC_HEAD:{label}:{depth:.2f}m",
                                        (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.5, (0, 0, 180), 1)

                    # 하단 좌/우
                    if cv2.pointPolygonTest(ground_left[0], (cx, ref_y), False) >= 0:
                        if label in static_ground_allowed:
                            if depth <= 2.0:
                                ground_left_state = "warning"
                            elif 2.0 < depth <= 3.0:
                                ground_left_state = "caution"
                            else:
                                ground_left_state = "safe"
                            cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 0, 180), 2)
                            cv2.putText(color_image, f"STATIC_GROUND:{label}:{depth:.2f}m",
                                        (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.5, (0, 0, 180), 1)

                    if cv2.pointPolygonTest(ground_right[0], (cx, ref_y), False) >= 0:
                        if label in static_ground_allowed:
                            if depth <= 2.0:
                                ground_right_state = "warning"
                            elif 2.0 < depth <= 3.0:
                                ground_right_state = "caution"
                            else:
                                ground_right_state = "safe"
                            cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 0, 180), 2)
                            cv2.putText(color_image, f"STATIC_GROUND:{label}:{depth:.2f}m",
                                        (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.5, (0, 0, 180), 1)

                # ------------------------------
                # Surface 처리 (노면 장애물, 하단 ROI만)
                # ------------------------------
                elif model_name == "Surface":
                    if label not in surface_ground_allowed:
                        continue
                    if cv2.pointPolygonTest(ground_left[0], (cx, ref_y), False) >= 0:
                        if depth <= 2.0:
                            ground_left_state = "warning"
                        elif 2.0 < depth <= 3.0:
                            ground_left_state = "caution"
                    elif cv2.pointPolygonTest(ground_right[0], (cx, ref_y), False) >= 0:
                        if depth <= 2.0:
                            ground_right_state = "warning"
                        elif 2.0 < depth <= 3.0:
                            ground_right_state = "caution"

                    cv2.rectangle(color_image, (x1, y1), (x2, y2), (180, 0, 180), 2)
                    cv2.putText(color_image, f"SURFACE:{label}:{depth:.2f}m",
                                (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (180, 0, 180), 1)

        # ------------------------------
        # ROI 시각화
        # ------------------------------
        overlay = color_image.copy()
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

        display = cv2.addWeighted(overlay, 0.35, color_image, 0.65, 0)
        cv2.polylines(display, ground_left, True, left_color, 2)
        cv2.polylines(display, ground_right, True, right_color, 2)
        cv2.polylines(display, head_roi, True, head_color, 2)

        cv2.putText(display,
                    f"Ground-L: {ground_left_state}  Ground-R: {ground_right_state}  Head: {head_state}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        cv2.imshow("Safety Monitor", display)
        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    if pipeline is not None:
        pipeline.stop()
    cv2.destroyAllWindows()