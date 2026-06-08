"""
ReaTrack API — FastAPI + PostgreSQL (Railway)
---------------------------------------------
Переписано з asyncpg → psycopg2 для сумісності з Railway.

asyncpg не підтримує параметр ?sslmode=require в URL та має обмеження
з кількома SQL-виразами в одному execute(). psycopg2 позбавлений цих проблем.

Endpoints:
  GET  /health        — перевірка живості + стан БД
  POST /register      — реєстрація
  POST /login         — вхід
  GET  /me            — поточний користувач (Bearer token)
  POST /analyze       — аналіз кадру через AI (base64 → MediaPipe)
  POST /sessions      — зберегти сесію (Bearer token)
  GET  /sessions      — список сесій (Bearer token)
  GET  /sessions/stats— агрегована статистика (Bearer token)
"""

import base64
import os
import tempfile
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

import psycopg2
import psycopg2.extras
import psycopg2.pool
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
import bcrypt as _bcrypt
from pydantic import BaseModel

# AI-модуль (pose_analyzer.py має лежати поруч з main.py)
from pose_analyzer import analyze_pose, JOINT_CONFIGS

# ── Конфіг ──────────────────────────────────────────────────────────────────
DATABASE_URL      = os.environ["DATABASE_URL"]
SECRET_KEY        = os.environ.get("SECRET_KEY", "змініть-у-продакшні")
ALGORITHM         = "HS256"
TOKEN_EXPIRE_DAYS = 30

# ── Ініціалізація ────────────────────────────────────────────────────────────
app = FastAPI(title="ReaTrack API", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()

# ── Connection pool (psycopg2) ───────────────────────────────────────────────
_pool: psycopg2.pool.ThreadedConnectionPool | None = None


@app.on_event("startup")
def startup() -> None:
    global _pool
    _pool = psycopg2.pool.ThreadedConnectionPool(1, 5, DATABASE_URL)

    with get_db() as conn:
        cur = conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id            TEXT PRIMARY KEY,
                email         TEXT UNIQUE NOT NULL,
                display_name  TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                created_at    TIMESTAMPTZ DEFAULT now()
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id            TEXT PRIMARY KEY,
                user_id       TEXT REFERENCES users(id) ON DELETE CASCADE,
                exercise_id   TEXT,
                exercise_name TEXT,
                total_reps    INT  DEFAULT 0,
                correct_reps  INT  DEFAULT 0,
                score         INT  DEFAULT 0,
                date          TIMESTAMPTZ DEFAULT now()
            )
        """)

        conn.commit()
        cur.close()


@app.on_event("shutdown")
def shutdown() -> None:
    if _pool:
        _pool.closeall()


@contextmanager
def get_db():
    """Бере з'єднання з пулу і повертає після використання."""
    conn = _pool.getconn()
    try:
        yield conn
    except Exception:
        conn.rollback()
        raise
    finally:
        _pool.putconn(conn)


# ── JWT хелпери ───────────────────────────────────────────────────────────────
def make_token(user_id: str) -> str:
    exp = datetime.now(timezone.utc) + timedelta(days=TOKEN_EXPIRE_DAYS)
    return jwt.encode({"sub": user_id, "exp": exp}, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user_id(
    creds: HTTPAuthorizationCredentials = Depends(security),
) -> str:
    try:
        payload = jwt.decode(creds.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        uid = payload.get("sub")
        if not uid:
            raise ValueError("no sub")
        return uid
    except (JWTError, ValueError):
        raise HTTPException(status_code=401, detail="Невалідний або прострочений токен.")


# ── Схеми запитів ────────────────────────────────────────────────────────────
class RegisterBody(BaseModel):
    email:        str
    password:     str
    display_name: str


class LoginBody(BaseModel):
    email:    str
    password: str


class SessionBody(BaseModel):
    exercise_id:   str
    exercise_name: str
    total_reps:    int
    correct_reps:  int
    score:         int


class AnalyzeBody(BaseModel):
    """
    Тіло запиту для POST /analyze.

    Поля:
      frame           — зображення у форматі base64 (JPEG або PNG).
                        Може містити data URI prefix: "data:image/jpeg;base64,..."
      joint           — суглоб для аналізу (напр. "left_elbow", "left_knee")
      reference_angle — еталонний кут у градусах (напр. 160.0)
      tolerance       — допустиме відхилення (за замовчуванням ±10°)
    """
    frame:           str
    joint:           str   = "left_elbow"
    reference_angle: float = 160.0
    tolerance:       float = 10.0


# ── Маршрути ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    try:
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute("SELECT 1")
            cur.close()
        return {"status": "ok", "db": "connected"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"DB error: {e}")


@app.post("/register", status_code=201)
def register(body: RegisterBody):
    with get_db() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        cur.execute("SELECT id FROM users WHERE email = %s", (body.email,))
        if cur.fetchone():
            cur.close()
            raise HTTPException(
                status_code=409,
                detail="Користувач із таким email вже існує.",
            )

        uid     = str(uuid.uuid4())
        pw_hash = _bcrypt.hashpw(body.password.encode(), _bcrypt.gensalt()).decode()

        cur.execute(
            """
            INSERT INTO users (id, email, display_name, password_hash)
            VALUES (%s, %s, %s, %s)
            """,
            (uid, body.email, body.display_name, pw_hash),
        )
        conn.commit()
        cur.close()

    return {
        "token": make_token(uid),
        "user":  {
            "uid":         uid,
            "email":       body.email,
            "displayName": body.display_name,
        },
    }


@app.post("/login")
def login(body: LoginBody):
    with get_db() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT * FROM users WHERE email = %s", (body.email,))
        row = cur.fetchone()
        cur.close()

    if not row or not _bcrypt.checkpw(body.password.encode(), row["password_hash"].encode()):
        raise HTTPException(status_code=401, detail="Невірний email або пароль.")

    return {
        "token": make_token(row["id"]),
        "user":  {
            "uid":         row["id"],
            "email":       row["email"],
            "displayName": row["display_name"],
        },
    }


@app.get("/me")
def me(uid: str = Depends(get_current_user_id)):
    with get_db() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            "SELECT id, email, display_name FROM users WHERE id = %s", (uid,)
        )
        row = cur.fetchone()
        cur.close()

    if not row:
        raise HTTPException(status_code=404, detail="Користувача не знайдено.")

    return {
        "uid":         row["id"],
        "email":       row["email"],
        "displayName": row["display_name"],
    }


# ── /analyze ─────────────────────────────────────────────────────────────────
@app.post("/analyze")
def analyze_frame(body: AnalyzeBody):
    """
    Аналізує кадр вправи через MediaPipe.

    Приймає зображення у форматі base64, передає його до pose_analyzer.py,
    і повертає результат аналізу кута суглоба.

    Приклад запиту:
        {
            "frame": "<base64-encoded JPEG>",
            "joint": "left_knee",
            "reference_angle": 90.0,
            "tolerance": 10.0
        }

    Відповідь (успіх):
        {
            "status": "ok",
            "joint": "left_knee",
            "measured_angle": 88.5,
            "reference_angle": 90.0,
            "deviation": 1.5,
            "tolerance": 10.0,
            "is_correct": true,
            "confidence": 0.94,
            "message": "Кут left_knee: 88.5° ..."
        }
    """
    # Валідація суглоба
    if body.joint not in JOINT_CONFIGS:
        valid = ", ".join(JOINT_CONFIGS.keys())
        raise HTTPException(
            status_code=422,
            detail=f"Невідомий суглоб '{body.joint}'. Доступні: {valid}",
        )

    # Декодуємо base64 → байти зображення
    try:
        raw_b64 = body.frame
        # Прибираємо data URI prefix якщо є (напр. "data:image/jpeg;base64,...")
        if "," in raw_b64:
            raw_b64 = raw_b64.split(",", 1)[1]
        image_bytes = base64.b64decode(raw_b64)
    except Exception:
        raise HTTPException(
            status_code=422,
            detail="Поле 'frame' містить некоректний base64.",
        )

    # Записуємо у тимчасовий файл (pose_analyzer потребує шляху до файлу)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False, dir="/tmp") as tmp:
            tmp.write(image_bytes)
            tmp_path = tmp.name

        result = analyze_pose(
            image_path=tmp_path,
            joint_name=body.joint,
            reference_angle=body.reference_angle,
            tolerance=body.tolerance,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Помилка AI-модуля: {exc}")
    finally:
        # Завжди видаляємо тимчасовий файл
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    # AI повернув помилку (поза не знайдена, поганий кадр тощо)
    if result.get("status") == "error":
        return {
            "status":     "error",
            "is_correct": False,
            "message":    result.get("message", "Невідома помилка AI."),
        }

    # is_error (pose_analyzer) → is_correct (зручніше для мобільного клієнта)
    return {
        "status":          "ok",
        "joint":           result["joint"],
        "landmarks_used":  result["landmarks_used"],
        "measured_angle":  result["measured_angle"],
        "reference_angle": result["reference_angle"],
        "deviation":       result["deviation"],
        "tolerance":       result["tolerance"],
        "is_correct":      not result["is_error"],
        "confidence":      result["confidence"],
        "message":         result["message"],
    }


# ── /sessions ─────────────────────────────────────────────────────────────────
@app.post("/sessions", status_code=201)
def create_session(body: SessionBody, uid: str = Depends(get_current_user_id)):
    with get_db() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        sid = str(uuid.uuid4())
        cur.execute(
            """
            INSERT INTO sessions
                (id, user_id, exercise_id, exercise_name, total_reps, correct_reps, score)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING id, exercise_name, total_reps, correct_reps, score, date
            """,
            (sid, uid, body.exercise_id, body.exercise_name,
             body.total_reps, body.correct_reps, body.score),
        )
        row = cur.fetchone()   # fetchone() перед commit() — RETURNING доступний одразу
        conn.commit()
        cur.close()

    total = (row["total_reps"] if row else body.total_reps)
    return {
        "id":             row["id"],
        "exerciseName":   row["exercise_name"],
        "totalReps":      row["total_reps"],
        "correctReps":    row["correct_reps"],
        "correctPercent": round(row["correct_reps"] / max(total, 1) * 100),
        "pointsEarned":   row["score"],
        "completedAt":    row["date"].isoformat(),
    }


@app.get("/sessions")
def get_sessions(uid: str = Depends(get_current_user_id)):
    with get_db() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            "SELECT * FROM sessions WHERE user_id = %s ORDER BY date DESC", (uid,)
        )
        rows = cur.fetchall()
        cur.close()

    return [
        {
            "id":             r["id"],
            "exerciseName":   r["exercise_name"],
            "totalReps":      r["total_reps"],
            "correctReps":    r["correct_reps"],
            "correctPercent": round(r["correct_reps"] / max(r["total_reps"], 1) * 100),
            "pointsEarned":   r["score"],
            "completedAt":    r["date"].isoformat(),
        }
        for r in rows
    ]


@app.get("/sessions/stats")
def get_stats(uid: str = Depends(get_current_user_id)):
    with get_db() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            """
            SELECT
                COUNT(*)::int                                    AS total_sessions,
                COALESCE(SUM(score), 0)::int                    AS total_points,
                COALESCE(ROUND(AVG(
                    CASE WHEN total_reps > 0
                         THEN correct_reps::float / total_reps * 100
                         ELSE 0
                    END
                )), 0)::int                                     AS avg_correct_percent
            FROM sessions
            WHERE user_id = %s
            """,
            (uid,),
        )
        row = cur.fetchone()
        cur.close()

    return {
        "totalSessions":     row["total_sessions"],
        "totalPoints":       row["total_points"],
        "avgCorrectPercent": row["avg_correct_percent"],
    }