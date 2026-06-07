"""
ReaTrack API — FastAPI + PostgreSQL backend (Railway)
-----------------------------------------------------
Endpoints:
  POST /register      → реєстрація
  POST /login         → вхід
  GET  /me            → поточний користувач (токен)
  POST /sessions      → зберегти сесію (токен)
  GET  /sessions      → список сесій (токен)
  GET  /sessions/stats→ агрегована статистика (токен)
  GET  /health        → перевірка живості
"""

import os
import uuid
from datetime import datetime, timedelta, timezone

import asyncpg
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

# ── Конфіг ──────────────────────────────────────────────────────────────────
DATABASE_URL      = os.environ["DATABASE_URL"]
SECRET_KEY        = os.environ.get("SECRET_KEY", "змініть-у-продакшні")
ALGORITHM         = "HS256"
TOKEN_EXPIRE_DAYS = 30

# ── Ініціалізація ────────────────────────────────────────────────────────────
app = FastAPI(title="ReaTrack API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

pwd_ctx  = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

pool: asyncpg.Pool | None = None

# ── Старт / зупинка ──────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    global pool
    pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=5)
    async with pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id            TEXT PRIMARY KEY,
                email         TEXT UNIQUE NOT NULL,
                display_name  TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                created_at    TIMESTAMPTZ DEFAULT now()
            );

            CREATE TABLE IF NOT EXISTS sessions (
                id            TEXT PRIMARY KEY,
                user_id       TEXT REFERENCES users(id) ON DELETE CASCADE,
                exercise_id   TEXT,
                exercise_name TEXT,
                total_reps    INT  DEFAULT 0,
                correct_reps  INT  DEFAULT 0,
                score         INT  DEFAULT 0,
                date          TIMESTAMPTZ DEFAULT now()
            );
        """)

@app.on_event("shutdown")
async def shutdown():
    await pool.close()

# ── JWT хелпери ───────────────────────────────────────────────────────────────
def make_token(user_id: str) -> str:
    exp = datetime.now(timezone.utc) + timedelta(days=TOKEN_EXPIRE_DAYS)
    return jwt.encode({"sub": user_id, "exp": exp}, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user_id(
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

# ── Маршрути ─────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/register", status_code=201)
async def register(body: RegisterBody):
    async with pool.acquire() as conn:
        existing = await conn.fetchrow(
            "SELECT id FROM users WHERE email = $1", body.email
        )
        if existing:
            raise HTTPException(
                status_code=409,
                detail="Користувач із таким email вже існує.",
            )

        uid     = str(uuid.uuid4())
        pw_hash = pwd_ctx.hash(body.password)

        await conn.execute(
            """
            INSERT INTO users (id, email, display_name, password_hash)
            VALUES ($1, $2, $3, $4)
            """,
            uid, body.email, body.display_name, pw_hash,
        )

    return {
        "token": make_token(uid),
        "user":  {
            "uid":         uid,
            "email":       body.email,
            "displayName": body.display_name,
        },
    }


@app.post("/login")
async def login(body: LoginBody):
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM users WHERE email = $1", body.email
        )

    if not row or not pwd_ctx.verify(body.password, row["password_hash"]):
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
async def me(uid: str = Depends(get_current_user_id)):
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id, email, display_name FROM users WHERE id = $1", uid
        )
    if not row:
        raise HTTPException(status_code=404, detail="Користувача не знайдено.")
    return {
        "uid":         row["id"],
        "email":       row["email"],
        "displayName": row["display_name"],
    }


@app.post("/sessions", status_code=201)
async def create_session(
    body: SessionBody,
    uid:  str = Depends(get_current_user_id),
):
    async with pool.acquire() as conn:
        sid = str(uuid.uuid4())
        row = await conn.fetchrow(
            """
            INSERT INTO sessions
                (id, user_id, exercise_id, exercise_name, total_reps, correct_reps, score)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            RETURNING id, exercise_name, total_reps, correct_reps, score, date
            """,
            sid, uid,
            body.exercise_id, body.exercise_name,
            body.total_reps,  body.correct_reps, body.score,
        )

    total = row["total_reps"]
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
async def get_sessions(uid: str = Depends(get_current_user_id)):
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT * FROM sessions WHERE user_id = $1 ORDER BY date DESC", uid
        )

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
async def get_stats(uid: str = Depends(get_current_user_id)):
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT
                COUNT(*)::int                                          AS total_sessions,
                COALESCE(SUM(score), 0)::int                           AS total_points,
                COALESCE(ROUND(AVG(
                    CASE WHEN total_reps > 0
                         THEN correct_reps::float / total_reps * 100
                         ELSE 0
                    END
                )), 0)::int                                            AS avg_correct_percent
            FROM sessions
            WHERE user_id = $1
            """,
            uid,
        )

    return {
        "totalSessions":     row["total_sessions"],
        "totalPoints":       row["total_points"],
        "avgCorrectPercent": row["avg_correct_percent"],
    }
