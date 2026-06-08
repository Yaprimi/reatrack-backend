"""
pose_analyzer.py — ReaTrack MVP: Pose Angle Analysis via MediaPipe
=================================================================
Приймає один кадр (зображення), знаходить ключові точки тіла (Pose Landmarks),
рахує кути між трьома точками за допомогою арктангенсу векторів (atan2).
Повертає JSON з результатом і прапорцем is_error: true, якщо кут поза межами ±10°.

Використання (CLI):
    python pose_analyzer.py --image frame.jpg --joint left_elbow --reference 160
    python pose_analyzer.py --image squat.png --joint left_knee  --reference 90 --save debug.jpg
    python pose_analyzer.py --image frame.jpg --config rehab_config.json --pretty

Використання (як модуль):
    from pose_analyzer import analyze_pose
    result = analyze_pose("frame.jpg", joint_name="left_elbow", reference_angle=160.0)
    print(result["is_error"])  # True / False

Залежності:
    pip install mediapipe>=0.10.0 opencv-python-headless>=4.8.0 numpy>=1.24.0
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
    "nose":             0,
    "left_shoulder":   11,  "right_shoulder":  12,
    "left_elbow":      13,  "right_elbow":     14,
    "left_wrist":      15,  "right_wrist":     16,
    "left_hip":        23,  "right_hip":       24,
    "left_knee":       25,  "right_knee":      26,
    "left_ankle":      27,  "right_ankle":     28,
    "left_heel":       29,  "right_heel":      30,
    "left_foot":       31,  "right_foot":      32,
}

# ── Конфігурації суглобів: (точка_A — вершина_B — точка_C) ──────────────────
# Кут вимірюється у вершині B (середня точка триплету).
JOINT_CONFIGS = {
    # Рука — лікоть = вершина (плече → лікоть → зап'ястя)
    "left_elbow":  ("left_shoulder",  "left_elbow",  "left_wrist"),
    "right_elbow": ("right_shoulder", "right_elbow", "right_wrist"),

    # Нога — коліно = вершина (стегно → коліно → щиколотка)
    "left_knee":   ("left_hip",   "left_knee",  "left_ankle"),
    "right_knee":  ("right_hip",  "right_knee", "right_ankle"),

    # Стегно — тазостегновий суглоб = вершина (плече → стегно → коліно)
    "left_hip":    ("left_shoulder",  "left_hip",  "left_knee"),
    "right_hip":   ("right_shoulder", "right_hip", "right_knee"),
}

# Допустиме відхилення від еталону (градуси)
ERROR_TOLERANCE_DEG: float = 10.0


# ── Геометрія ─────────────────────────────────────────────────────────────────

def calculate_angle(a: tuple, b: tuple, c: tuple) -> float:
    """
    Обчислює кут у вершині B між векторами BA та BC у градусах.

    Метод: арктангенс двох векторів через math.atan2().
    - atan2(y, x) повертає кут вектора відносно осі X (від -π до π).
    - Різниця кутів двох векторів = кут між ними.
    - Нормалізація до [0°, 180°] гарантує невід'ємний результат.

    Перевага над dot-product:
    - Стабільний при паралельних/антипаралельних векторах (немає ділення на нуль).
    - Точніший для малих і великих кутів.

    Args:
        a: (x, y) — перша точка (наприклад, плече).
        b: (x, y) — вершина кута (наприклад, лікоть).
        c: (x, y) — третя точка (наприклад, зап'ястя).

    Returns:
        Кут у градусах, округлений до 2 знаків після коми. Діапазон: [0.0, 180.0].
    """
    # Вектор BA (від вершини до точки A)
    bax = a[0] - b[0]
    bay = a[1] - b[1]

    # Вектор BC (від вершини до точки C)
    bcx = c[0] - b[0]
    bcy = c[1] - b[1]

    # Кут кожного вектора відносно горизонтальної осі X
    angle_ba = math.atan2(bay, bax)   # радіани, [-π, π]
    angle_bc = math.atan2(bcy, bcx)   # радіани, [-π, π]

    # Кут між векторами (різниця кутів)
    diff_deg = math.degrees(angle_ba - angle_bc)

    # Нормалізація: повертаємо найменший позитивний кут ≤ 180°
    diff_deg = abs(diff_deg) % 360.0
    if diff_deg > 180.0:
        diff_deg = 360.0 - diff_deg

    return round(diff_deg, 2)


def check_error(
    measured: float,
    reference: float,
    tolerance: float = ERROR_TOLERANCE_DEG,
) -> bool:
    """
    Повертає True, якщо виміряний кут виходить за межі [reference ± tolerance].

    Args:
        measured:  Виміряний кут (градуси).
        reference: Еталонний кут (градуси).
        tolerance: Допустиме відхилення (за замовчуванням ±10°).

    Returns:
        True  → помилка (кут поза нормою).
        False → кут у нормі.
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
    Аналізує позу на зображенні за допомогою MediaPipe Pose.

    Кроки:
    1. Завантажує зображення через OpenCV.
    2. Передає RGB-кадр до MediaPipe Pose (static_image_mode=True).
    3. Отримує координати трьох ключових точок для заданого суглоба.
    4. Обчислює кут через atan2-метод (calculate_angle).
    5. Порівнює з еталоном і повертає прапорець is_error.

    Args:
        image_path:               Шлях до зображення (JPG / PNG / BMP).
        joint_name:               Назва суглоба з JOINT_CONFIGS.
        reference_angle:          Еталонний кут (градуси).
        tolerance:                Допустиме відхилення (за замовчуванням ±10°).
        min_detection_confidence: Мінімальна впевненість MediaPipe (0.0–1.0).
        save_annotated:           Якщо вказано — шлях для збереження зображення з розміткою.

    Returns:
        dict з ключами:
            status          — "ok" | "error"
            joint           — назва суглоба
            landmarks_used  — [name_A, name_B, name_C]
            measured_angle  — виміряний кут (float)
            reference_angle — еталонний кут (float)
            deviation       — |measured - reference| (float)
            tolerance       — допустиме відхилення (float)
            is_error        — True, якщо deviation > tolerance
            confidence      — середня видимість трьох точок (0.0–1.0)
            message         — текстовий опис результату
    """
    # ── Валідація вхідних даних ───────────────────────────────────────────────
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
        return _error_response(
            f"Не вдалося відкрити зображення: {image_path}. "
            "Переконайтесь, що файл є валідним JPEG/PNG/BMP."
        )

    # ── Перевірка на чорний / занадто темний кадр ────────────────────────────
    # MediaPipe може повертати «landmarks» навіть на чорному зображенні,
    # тому перевіряємо середню яскравість ДО виклику pose.process().
    # Поріг 15 / 255 (~6 %) відсікає чорний екран і закриту лінзу,
    # але не зачіпає нормально освітлені сцени навіть у напівтемряві.
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(np.mean(gray))
    if mean_brightness < 15.0:
        return _error_response(
            f"Зображення занадто темне (яскравість={mean_brightness:.1f}/255). "
            "Перевірте освітлення або переконайтесь, що камера не заблокована."
        )

    # MediaPipe очікує RGB, OpenCV читає BGR
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # ── MediaPipe Pose (статичний режим для одного кадру) ─────────────────────
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(
        static_image_mode=True,           # один кадр — без відеотреку
        model_complexity=1,               # 0=lite, 1=full, 2=heavy
        enable_segmentation=False,        # не потрібна для кутів
        min_detection_confidence=min_detection_confidence,
    ) as pose:
        results = pose.process(image_rgb)

    if not results.pose_landmarks:
        return _error_response(
            "MediaPipe не виявив жодної точки тіла на зображенні. "
            "Переконайтесь, що людина повністю видима на кадрі "
            "та освітлення достатнє."
        )

    landmarks = results.pose_landmarks.landmark

    # ── Отримання трьох точок суглоба ────────────────────────────────────────
    name_a, name_b, name_c = JOINT_CONFIGS[joint_name]
    lm_a = landmarks[LANDMARK[name_a]]
    lm_b = landmarks[LANDMARK[name_b]]
    lm_c = landmarks[LANDMARK[name_c]]

    # Перевірка видимості — MediaPipe дає 0.0–1.0
    # Поріг підвищено з 0.3 → 0.5: при темному або розмитому кадрі MediaPipe
    # часто повертає «примарні» точки з visibility 0.3–0.49, що призводило
    # до хибно-позитивного результату «кут у нормі» на чорному екрані.
    avg_visibility = round(
        (lm_a.visibility + lm_b.visibility + lm_c.visibility) / 3.0, 3
    )
    if avg_visibility < 0.5:
        return _error_response(
            f"Ключові точки '{name_a}', '{name_b}', '{name_c}' погано видимі "
            f"(visibility={avg_visibility:.2f}). Спробуйте інший ракурс або кадр."
        )

    # Нормалізовані координати (0.0–1.0 відносно розміру кадру)
    pt_a = (lm_a.x, lm_a.y)
    pt_b = (lm_b.x, lm_b.y)
    pt_c = (lm_c.x, lm_c.y)

    # ── Обчислення кута (реальна тригонометрія, не симуляція) ─────────────────
    measured  = calculate_angle(pt_a, pt_b, pt_c)
    deviation = round(abs(measured - reference_angle), 2)
    has_error = check_error(measured, reference_angle, tolerance)

    # ── (Опційно) Збереження зображення з розміткою ───────────────────────────
    if save_annotated:
        _draw_and_save(
            image_bgr=image_bgr,
            landmarks=landmarks,
            pt_a=pt_a, pt_b=pt_b, pt_c=pt_c,
            name_a=name_a, name_b=name_b, name_c=name_c,
            measured=measured,
            reference=reference_angle,
            has_error=has_error,
            out_path=save_annotated,
        )

    # ── Формування відповіді ──────────────────────────────────────────────────
    if not has_error:
        verdict = "в нормі ✓"
    elif measured > reference_angle:
        verdict = f"завеликий (+{deviation}°) ✗"
    else:
        verdict = f"замалий (-{deviation}°) ✗"

    return {
        "status":          "ok",
        "joint":           joint_name,
        "landmarks_used":  [name_a, name_b, name_c],
        "measured_angle":  measured,
        "reference_angle": reference_angle,
        "deviation":       deviation,
        "tolerance":       tolerance,
        "is_error":        has_error,
        "confidence":      avg_visibility,
        "message": (
            f"Кут {name_b}: {measured}° "
            f"(еталон {reference_angle}° ±{tolerance}°) — {verdict}"
        ),
    }


# ── Допоміжні функції ─────────────────────────────────────────────────────────

def _error_response(message: str) -> dict:
    """Формує стандартну відповідь про помилку."""
    return {
        "status":   "error",
        "is_error": True,
        "message":  message,
    }


def _draw_and_save(
    image_bgr, landmarks,
    pt_a: tuple, pt_b: tuple, pt_c: tuple,
    name_a: str, name_b: str, name_c: str,
    measured: float, reference: float,
    has_error: bool, out_path: str,
) -> None:
    """
    Малює три ключові точки, з'єднувальні лінії та кут на кадрі.
    Зберігає результат у файл out_path.
    """
    h, w = image_bgr.shape[:2]
    frame = image_bgr.copy()

    # Кольори: зелений = норма, червоний = помилка
    ok_color  = (0,  200,   0)
    err_color = (0,   0,  220)
    line_color = err_color if has_error else ok_color

    # Піксельні координати
    pa = (int(pt_a[0] * w), int(pt_a[1] * h))
    pb = (int(pt_b[0] * w), int(pt_b[1] * h))
    pc = (int(pt_c[0] * w), int(pt_c[1] * h))

    # Лінії між точками
    cv2.line(frame, pa, pb, line_color, 2, cv2.LINE_AA)
    cv2.line(frame, pb, pc, line_color, 2, cv2.LINE_AA)

    # Кола на точках
    for px_pt, name, color in [
        (pa, name_a, (255, 165, 0)),
        (pb, name_b, err_color if has_error else ok_color),
        (pc, name_c, (255, 165, 0)),
    ]:
        cv2.circle(frame, px_pt, 8, color, -1, cv2.LINE_AA)
        label = name.replace("_", " ")
        cv2.putText(frame, label, (px_pt[0] + 10, px_pt[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    # Текстовий оверлей
    status_txt = f"{'ПОМИЛКА' if has_error else 'OK'}: {measured}\xb0"
    cv2.putText(frame, status_txt, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                err_color if has_error else ok_color, 2, cv2.LINE_AA)
    cv2.putText(frame,
                f"Еталон: {reference}\xb0  \xb110\xb0",
                (20, 76),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (60, 60, 60), 1, cv2.LINE_AA)

    cv2.imwrite(out_path, frame)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ReaTrack MVP: аналіз кута суглоба через MediaPipe",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Приклади:
  python pose_analyzer.py --image frame.jpg --joint left_elbow --reference 160
  python pose_analyzer.py --image squat.png --joint left_knee  --reference 90 --save debug.jpg
  python pose_analyzer.py --image frame.jpg --config rehab_config.json --pretty

Доступні суглоби:
  left_elbow,  right_elbow   (плече – лікоть – зап'ястя)
  left_knee,   right_knee    (стегно – коліно – щиколотка)
  left_hip,    right_hip     (плече – стегно – коліно)
        """,
    )
    p.add_argument("--image",      required=True, help="Шлях до зображення (JPG/PNG)")
    p.add_argument("--joint",      default="left_elbow",
                   choices=list(JOINT_CONFIGS.keys()),
                   help="Суглоб для аналізу (default: left_elbow)")
    p.add_argument("--reference",  type=float, default=160.0,
                   help="Еталонний кут у градусах (default: 160)")
    p.add_argument("--tolerance",  type=float, default=ERROR_TOLERANCE_DEG,
                   help=f"Допустиме відхилення ±° (default: {ERROR_TOLERANCE_DEG})")
    p.add_argument("--config",     help="JSON-файл з параметрами (замість --joint/--reference)")
    p.add_argument("--save",       help="Зберегти зображення з розміткою (вказати .jpg/.png)")
    p.add_argument("--confidence", type=float, default=0.5,
                   help="Мінімальна впевненість MediaPipe (default: 0.5)")
    p.add_argument("--pretty",     action="store_true",
                   help="JSON-вивід з відступами (читабельний формат)")
    return p


def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    joint     = args.joint
    reference = args.reference
    tolerance = args.tolerance

    # Завантаження параметрів з JSON-конфігу (пріоритет над CLI-флагами)
    if args.config:
        try:
            with open(args.config, encoding="utf-8") as f:
                cfg = json.load(f)
            joint     = cfg.get("joint",     joint)
            reference = float(cfg.get("reference", reference))
            tolerance = float(cfg.get("tolerance", tolerance))
        except Exception as exc:
            print(json.dumps({"status": "error", "is_error": True,
                              "message": f"Помилка конфігу: {exc}"},
                             ensure_ascii=False))
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

    # Exit-код 1, якщо is_error=True (зручно для shell-скриптів і CI)
    sys.exit(1 if result.get("is_error") else 0)


if __name__ == "__main__":
    main()