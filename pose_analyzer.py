"""
pose_analyzer.py — ReaTrack MVP: Pose Angle Analysis via MediaPipe
=================================================================
Приймає один кадр (зображення), знаходить ключові точки тіла (Pose Landmarks),
рахує кути між трьома точками за допомогою арктангенсу векторів.
Повертає JSON з результатом і прапорцем is_error: true, якщо кут поза межами ±10°.

Використання:
    python pose_analyzer.py --image frame.jpg --joint left_elbow --reference 160

    # Або через JSON-конфіг:
    python pose_analyzer.py --image frame.jpg --config config.json

    # Або як модуль:
    from pose_analyzer import analyze_pose
    result = analyze_pose("frame.jpg", joint_name="left_elbow", reference_angle=160.0)
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np


# ── MediaPipe Pose Landmark індекси ──────────────────────────────────────────
# Повний список: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
LANDMARK = {
    # Торс
    "nose": 0,
    "left_shoulder": 11,  "right_shoulder": 12,
    "left_elbow":    13,  "right_elbow":    14,
    "left_wrist":    15,  "right_wrist":    16,
    "left_hip":      23,  "right_hip":      24,
    "left_knee":     25,  "right_knee":     26,
    "left_ankle":    27,  "right_ankle":    28,
    "left_heel":     29,  "right_heel":     30,
    "left_foot":     31,  "right_foot":     32,
}

# ── Еталонні конфігурації суглобів (точка_A — вершина_B — точка_C) ───────────
JOINT_CONFIGS = {
    # Рука (лікоть = вершина кута)
    "left_elbow":  ("left_shoulder",  "left_elbow",  "left_wrist"),
    "right_elbow": ("right_shoulder", "right_elbow", "right_wrist"),

    # Нога (коліно = вершина кута)
    "left_knee":   ("left_hip",  "left_knee",  "left_ankle"),
    "right_knee":  ("right_hip", "right_knee", "right_ankle"),

    # Стегно (стегно = вершина)
    "left_hip":    ("left_shoulder",  "left_hip",  "left_knee"),
    "right_hip":   ("right_shoulder", "right_hip", "right_knee"),
}

ERROR_TOLERANCE_DEG = 10.0   # ±10° — допустиме відхилення від еталону


# ── Геометрія ─────────────────────────────────────────────────────────────────

def _landmark_to_xy(lm) -> tuple[float, float]:
    """Повертає (x, y) нормалізовані координати точки MediaPipe."""
    return lm.x, lm.y


def calculate_angle(a: tuple, b: tuple, c: tuple) -> float:
    """
    Рахує кут у вершині B (між векторами BA і BC) у градусах.

    Алгоритм — арктангенс двох векторів (atan2), щоб отримати
    знаковий кут і уникнути виродження при паралельних векторах.

    Args:
        a: (x, y) — початкова точка
        b: (x, y) — вершина кута
        c: (x, y) — кінцева точка

    Returns:
        Кут у градусах [0°, 180°]
    """
    ax, ay = a[0] - b[0], a[1] - b[1]   # вектор BA
    cx, cy = c[0] - b[0], c[1] - b[1]   # вектор BC

    # atan2 дає кут кожного вектора відносно осі X
    angle_a = math.atan2(ay, ax)
    angle_c = math.atan2(cy, cx)

    # Різниця кутів → кут між векторами
    angle_deg = math.degrees(angle_a - angle_c)

    # Нормалізуємо до [0°, 360°] і беремо найменший кут ≤ 180°
    angle_deg = abs(angle_deg) % 360
    if angle_deg > 180:
        angle_deg = 360 - angle_deg

    return round(angle_deg, 2)


def check_error(measured: float, reference: float,
                tolerance: float = ERROR_TOLERANCE_DEG) -> bool:
    """
    Повертає True, якщо виміряний кут виходить за межі
    [reference - tolerance, reference + tolerance].
    """
    return abs(measured - reference) > tolerance


# ── Основна функція аналізу ───────────────────────────────────────────────────

def analyze_pose(
    image_path: str,
    joint_name: str = "left_elbow",
    reference_angle: float = 160.0,
    tolerance: float = ERROR_TOLERANCE_DEG,
    min_detection_confidence: float = 0.5,
    save_annotated: Optional[str] = None,
) -> dict:
    """
    Головна функція: аналізує позу на зображенні та повертає результат.

    Args:
        image_path:                Шлях до зображення (JPG / PNG / BMP).
        joint_name:                Назва суглоба з JOINT_CONFIGS.
        reference_angle:           Еталонний кут у градусах.
        tolerance:                 Допустиме відхилення (за замовчуванням ±10°).
        min_detection_confidence:  Мінімальна впевненість MediaPipe (0–1).
        save_annotated:            Якщо вказано, зберігає зображення з розміткою.

    Returns:
        dict з ключами:
            status          — "ok" або "error"
            joint           — назва суглоба
            measured_angle  — виміряний кут (градуси)
            reference_angle — еталонний кут
            deviation       — відхилення від еталону
            tolerance       — допустиме відхилення
            is_error        — true, якщо відхилення > tolerance
            landmarks_used  — назви трьох точок
            confidence      — видимість ключових точок (0–1)
            message         — текстовий опис результату
    """
    path = Path(image_path)
    if not path.exists():
        return _error_response(f"Файл не знайдено: {image_path}")

    if joint_name not in JOINT_CONFIGS:
        valid = ", ".join(JOINT_CONFIGS.keys())
        return _error_response(
            f"Невідомий суглоб '{joint_name}'. Доступні: {valid}"
        )

    # ── Завантаження зображення ───────────────────────────────────────────────
    image_bgr = cv2.imread(str(path))
    if image_bgr is None:
        return _error_response(f"Не вдалося відкрити зображення: {image_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # ── Ініціалізація MediaPipe Pose ──────────────────────────────────────────
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,                        # 0=lite, 1=full, 2=heavy
        min_detection_confidence=min_detection_confidence,
    ) as pose:
        results = pose.process(image_rgb)

    if not results.pose_landmarks:
        return _error_response(
            "MediaPipe не виявив жодної точки тіла на зображенні. "
            "Переконайтесь, що людина видима на кадрі."
        )

    landmarks = results.pose_landmarks.landmark

    # ── Отримання трьох точок суглоба ────────────────────────────────────────
    name_a, name_b, name_c = JOINT_CONFIGS[joint_name]
    idx_a, idx_b, idx_c = (LANDMARK[n] for n in (name_a, name_b, name_c))

    lm_a = landmarks[idx_a]
    lm_b = landmarks[idx_b]
    lm_c = landmarks[idx_c]

    # Середня видимість трьох точок
    avg_visibility = round(
        (lm_a.visibility + lm_b.visibility + lm_c.visibility) / 3, 3
    )

    if avg_visibility < 0.3:
        return _error_response(
            f"Ключові точки '{name_a}', '{name_b}', '{name_c}' погано видимі "
            f"(visibility={avg_visibility:.2f}). Спробуйте інший кадр."
        )

    pt_a = _landmark_to_xy(lm_a)
    pt_b = _landmark_to_xy(lm_b)
    pt_c = _landmark_to_xy(lm_c)

    # ── Розрахунок кута ───────────────────────────────────────────────────────
    measured = calculate_angle(pt_a, pt_b, pt_c)
    deviation = round(abs(measured - reference_angle), 2)
    has_error = check_error(measured, reference_angle, tolerance)

    # ── (Опційно) Збереження зображення з розміткою ───────────────────────────
    if save_annotated:
        _draw_and_save(
            image_bgr, landmarks, pt_a, pt_b, pt_c,
            name_a, name_b, name_c,
            measured, reference_angle, has_error,
            save_annotated,
        )

    # ── Формування відповіді ──────────────────────────────────────────────────
    direction = "в нормі" if not has_error else (
        "завеликий" if measured > reference_angle else "замалий"
    )

    return {
        "status": "ok",
        "joint": joint_name,
        "landmarks_used": [name_a, name_b, name_c],
        "measured_angle": measured,
        "reference_angle": reference_angle,
        "deviation": deviation,
        "tolerance": tolerance,
        "is_error": has_error,
        "confidence": avg_visibility,
        "message": (
            f"Кут {name_b}: {measured}° (еталон {reference_angle}°, "
            f"відхилення {deviation}° — {direction})"
        ),
    }


# ── Допоміжні функції ─────────────────────────────────────────────────────────

def _error_response(message: str) -> dict:
    return {
        "status": "error",
        "is_error": True,
        "message": message,
    }


def _draw_and_save(
    image_bgr, landmarks, pt_a, pt_b, pt_c,
    name_a, name_b, name_c,
    measured, reference, has_error, out_path
):
    """Малює скелет і кут на зображенні, зберігає файл."""
    h, w = image_bgr.shape[:2]
    annotated = image_bgr.copy()

    # Малюємо весь скелет через mp.solutions.drawing_utils
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    # Скелет (сірий, тонкий)
    skeleton_style = mp_drawing.DrawingSpec(color=(180, 180, 180), thickness=1)
    landmark_style = mp_drawing.DrawingSpec(color=(0, 200, 255), thickness=2, circle_radius=3)

    # Малюємо три ключові точки великими колами
    for pt, name, color in [
        (pt_a, name_a, (255, 165, 0)),
        (pt_b, name_b, (0, 0, 255) if has_error else (0, 255, 0)),
        (pt_c, name_c, (255, 165, 0)),
    ]:
        px, py = int(pt[0] * w), int(pt[1] * h)
        cv2.circle(annotated, (px, py), 8, color, -1)
        cv2.putText(annotated, name.split("_")[1], (px + 10, py),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # Лінії між точками
    pa = (int(pt_a[0] * w), int(pt_a[1] * h))
    pb = (int(pt_b[0] * w), int(pt_b[1] * h))
    pc = (int(pt_c[0] * w), int(pt_c[1] * h))
    line_color = (0, 0, 220) if has_error else (0, 220, 0)
    cv2.line(annotated, pa, pb, line_color, 2)
    cv2.line(annotated, pb, pc, line_color, 2)

    # Текст з кутом
    status_text = f"ПОМИЛКА: {measured}deg" if has_error else f"OK: {measured}deg"
    text_color = (0, 0, 220) if has_error else (0, 180, 0)
    cv2.putText(annotated, status_text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, text_color, 2, cv2.LINE_AA)
    cv2.putText(annotated, f"Еталон: {reference}deg  +/-10deg", (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 80), 1, cv2.LINE_AA)

    cv2.imwrite(out_path, annotated)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ReaTrack MVP: Аналіз кута суглоба через MediaPipe",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Приклади:
  python pose_analyzer.py --image frame.jpg --joint left_elbow --reference 160
  python pose_analyzer.py --image squat.png --joint left_knee  --reference 90 --save debug.jpg
  python pose_analyzer.py --image frame.jpg --config rehab_config.json

Доступні суглоби:
  left_elbow, right_elbow   (плече–лікоть–зап'ястя)
  left_knee,  right_knee    (стегно–коліно–щиколотка)
  left_hip,   right_hip     (плече–стегно–коліно)
        """
    )
    p.add_argument("--image",     required=True,  help="Шлях до зображення")
    p.add_argument("--joint",     default="left_elbow",
                   choices=list(JOINT_CONFIGS.keys()),
                   help="Суглоб для аналізу (default: left_elbow)")
    p.add_argument("--reference", type=float, default=160.0,
                   help="Еталонний кут у градусах (default: 160)")
    p.add_argument("--tolerance", type=float, default=ERROR_TOLERANCE_DEG,
                   help=f"Допустиме відхилення в градусах (default: {ERROR_TOLERANCE_DEG})")
    p.add_argument("--config",    help="JSON-файл з параметрами (замість --joint/--reference)")
    p.add_argument("--save",      help="Зберегти зображення з розміткою (шлях до .jpg)")
    p.add_argument("--confidence", type=float, default=0.5,
                   help="Мінімальна впевненість MediaPipe (default: 0.5)")
    p.add_argument("--pretty", action="store_true",
                   help="Красивий JSON-вивід з відступами")
    return p


def main():
    parser = _build_parser()
    args = parser.parse_args()

    # Завантаження параметрів з JSON-конфігу (якщо вказано)
    joint     = args.joint
    reference = args.reference
    tolerance = args.tolerance

    if args.config:
        try:
            with open(args.config) as f:
                cfg = json.load(f)
            joint     = cfg.get("joint",     joint)
            reference = cfg.get("reference", reference)
            tolerance = cfg.get("tolerance", tolerance)
        except Exception as e:
            print(json.dumps({"status": "error", "is_error": True,
                              "message": f"Помилка конфігу: {e}"}))
            sys.exit(1)

    result = analyze_pose(
        image_path=args.image,
        joint_name=joint,
        reference_angle=reference,
        tolerance=tolerance,
        min_detection_confidence=args.confidence,
        save_annotated=args.save,
    )

    indent = 2 if args.pretty else None
    print(json.dumps(result, ensure_ascii=False, indent=indent))

    # Exit code: 1 якщо is_error = true (зручно для shell-скриптів)
    sys.exit(1 if result.get("is_error") else 0)


if __name__ == "__main__":
    main()
