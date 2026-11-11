"""Enterprise Operations & AI Support Platform: tickets, AI triage, knowledge base,
automation, analytics, activity log, and role-based login."""
import hashlib
import os
import random
import re
import secrets
from collections import Counter
from datetime import datetime, timedelta
from typing import Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import DateTime, String, Text, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column
from starlette.middleware.sessions import SessionMiddleware

load_dotenv()
import ai  # noqa: E402

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./tickets.db")
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
)

PRIORITIES = ["Low", "Medium", "High"]
STATUSES = ["To Do", "In Progress", "Done"]
AGENTS = ["Alex Rivera", "Priya Shah", "Sam Okafor", "Unassigned"]
# Automation rules
ESCALATION_HOURS = int(os.getenv("ESCALATION_HOURS", "2"))
CATEGORY_OWNER = {
    "Hardware": "Sam Okafor", "Software": "Priya Shah", "Network": "Alex Rivera",
    "Email": "Priya Shah", "Security": "Alex Rivera", "Accounts & Access": "Sam Okafor",
    "Cloud & Servers": "Alex Rivera",
}


class Base(DeclarativeBase):
    pass


class Ticket(Base):
    __tablename__ = "tickets"
    id: Mapped[int] = mapped_column(primary_key=True)
    subject: Mapped[str] = mapped_column(String(200))
    description: Mapped[str] = mapped_column(Text, default="")
    company: Mapped[str] = mapped_column(String(100))
    priority: Mapped[str] = mapped_column(String(10), default="Medium")
    status: Mapped[str] = mapped_column(String(20), default="To Do")
    assignee: Mapped[str] = mapped_column(String(100), default="Unassigned")
    category: Mapped[str] = mapped_column(String(40), default="Uncategorized")
    ai_summary: Mapped[str] = mapped_column(Text, default="")
    resolution: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    resolved_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    escalated_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class Article(Base):
    __tablename__ = "articles"
    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(200))
    category: Mapped[str] = mapped_column(String(40), default="Other")
    content: Mapped[str] = mapped_column(Text, default="")


class Activity(Base):
    __tablename__ = "activity"
    id: Mapped[int] = mapped_column(primary_key=True)
    ticket_id: Mapped[Optional[int]] = mapped_column(nullable=True)
    actor: Mapped[str] = mapped_column(String(100))
    action: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(50), unique=True)
    name: Mapped[str] = mapped_column(String(100))
    role: Mapped[str] = mapped_column(String(20))  # "agent" or "manager"
    password_hash: Mapped[str] = mapped_column(String(200))


# ---------- helpers ----------
def hash_password(pw: str, salt: Optional[str] = None) -> str:
    salt = salt or secrets.token_hex(8)
    digest = hashlib.pbkdf2_hmac("sha256", pw.encode(), salt.encode(), 100_000).hex()
    return f"{salt}${digest}"


def verify_password(pw: str, stored: str) -> bool:
    salt = stored.split("$")[0]
    return secrets.compare_digest(hash_password(pw, salt), stored)


def get_db():
    with Session(engine) as db:
        yield db


def current_user(request: Request, db: Session = Depends(get_db)) -> User:
    uid = request.session.get("uid")
    user = db.get(User, uid) if uid else None
    if not user:
        raise HTTPException(401, "Please sign in")
    return user


def manager_only(user: User = Depends(current_user)) -> User:
    if user.role != "manager":
        raise HTTPException(403, "Only managers can do this")
    return user


def log(db: Session, ticket_id, actor: str, action: str):
    db.add(Activity(ticket_id=ticket_id, actor=actor, action=action))


def iso(d):
    return d.isoformat() if d else None


def out(t: Ticket):
    return {
        "id": t.id, "subject": t.subject, "description": t.description,
        "company": t.company, "priority": t.priority, "status": t.status,
        "assignee": t.assignee, "category": t.category, "ai_summary": t.ai_summary,
        "resolution": t.resolution, "created_at": iso(t.created_at),
        "resolved_at": iso(t.resolved_at), "escalated_at": iso(t.escalated_at),
        "escalated": bool(t.escalated_at and t.status != "Done"),
    }


def check(priority: Optional[str], status: Optional[str]):
    if priority is not None and priority not in PRIORITIES:
        raise HTTPException(422, f"Priority must be one of {PRIORITIES}")
    if status is not None and status not in STATUSES:
        raise HTTPException(422, f"Status must be one of {STATUSES}")


STOP = set("the and for with not from that this are was has have new need needs request "
           "after before every into over".split())


def tokens(text: str) -> set:
    return {w for w in re.findall(r"[a-z0-9]{3,}", text.lower()) if w not in STOP}


def sim(a: set, b: set) -> float:
    return len(a & b) / ((len(a) * len(b)) ** 0.5) if a and b else 0.0


def similar_for(db: Session, t: Ticket):
    base = tokens(f"{t.subject} {t.description}")
    arts = sorted(((sim(base, tokens(a.title + " " + a.content)), a) for a in db.scalars(select(Article))),
                  key=lambda x: -x[0])
    past = sorted(((sim(base, tokens(o.subject + " " + o.resolution)), o) for o in db.scalars(
        select(Ticket).where(Ticket.status == "Done", Ticket.id != t.id))), key=lambda x: -x[0])
    return ([a for s, a in arts if s > 0.05][:3], [o for s, o in past if s > 0.05][:3])


def run_automation(db: Session):
    """Escalate High-priority tickets that nobody has started within ESCALATION_HOURS."""
    cutoff = datetime.utcnow() - timedelta(hours=ESCALATION_HOURS)
    stale = db.scalars(select(Ticket).where(
        Ticket.status == "To Do", Ticket.priority == "High",
        Ticket.escalated_at.is_(None), Ticket.created_at < cutoff))
    changed = False
    for t in stale:
        t.escalated_at = datetime.utcnow()
        log(db, t.id, "Automation",
            f"Escalated to manager: High priority and not started after {ESCALATION_HOURS} hours")
        changed = True
    if changed:
        db.commit()


def apply_ai(db: Session, t: Ticket, result: dict, actor: str, set_priority: bool):
    t.category = result["category"]
    t.ai_summary = result["summary"]
    if set_priority:
        t.priority = result["priority"]
    log(db, t.id, "AI assistant", f"Classified as {t.category}, priority {t.priority}")
    owner = CATEGORY_OWNER.get(t.category)
    if owner and t.assignee == "Unassigned":
        t.assignee = owner
        log(db, t.id, "Automation", f"Auto-assigned to {owner} (rule for {t.category})")


# ---------- seed data ----------
SEED_TICKETS = [
    ("Annual workstation refresh: quote for 10 new laptops", "Hardware", "Quote approved; 10 laptops ordered and imaged with the standard build."),
    ("Azure cost review: monthly spend jumped 42%", "Cloud & Servers", "Found idle VMs and oversized disks; resized them and set budget alerts."),
    ("Migrate local Active Directory to Entra ID", "Accounts & Access", "Hybrid sync configured and users migrated in batches over a weekend."),
    ("VPN drops every 20 minutes for remote staff", "Network", "Firewall idle timeout was too short; raised it to 8 hours and updated the VPN client."),
    ("Shared mailbox not receiving external email", "Email", "Mailbox was restricted to internal senders; enabled external delivery."),
    ("Deploy SSO with MFA across all staff accounts", "Security", "SSO enabled with authenticator app MFA; staff enrolled using a short guide."),
    ("Backup job failing nightly on file server", "Cloud & Servers", "Backup agent was out of date and the target disk was full; updated it and cleared old backups."),
    ("Printer not scanning to email after migration", "Hardware", "Printer SMTP settings pointed to the old mail server; updated to the new relay."),
    ("New hire needs laptop, accounts and building access", "Accounts & Access", "Provisioned laptop, accounts and access card using the onboarding checklist."),
    ("Configure VLANs and firewall rules for guest Wi-Fi", "Network", "Created an isolated guest VLAN with internet-only firewall rules."),
    ("Outlook keeps asking for password", "Email", "Cleared stored credentials and recreated the Outlook profile; modern authentication re-enabled."),
    ("Disaster recovery test for accounting database", "Cloud & Servers", "Restored the latest backup to a test server; recovery took 38 minutes."),
    ("Exchange to Microsoft 365 migration: 130 mailboxes", "Email", "Mailboxes moved in batches with no data loss; old server decommissioned."),
    ("Ransomware warning on a staff workstation", "Security", "Machine isolated and scanned; false positive from a cracked-software installer, which was removed."),
    ("Software license audit before renewal", "Software", "Audit found 12 unused licenses; renewal reduced accordingly."),
    ("Slow performance on the client portal", "Software", "Database queries lacked an index; added the index and restarted the app pool."),
    ("Phone system voicemail-to-email not working", "Network", "Voicemail server could not sign in to the mail relay; credentials refreshed."),
    ("Request: second monitor for reception desk", "Hardware", "Monitor and docking cable supplied and installed."),
    ("Password reset portal returns an error", "Accounts & Access", "Portal certificate had expired; renewed it and restarted the service."),
    ("Patch management: 14 servers behind on updates", "Cloud & Servers", "Patched in a maintenance window; automatic update schedule enabled."),
    ("Conference room display will not connect", "Hardware", "Replaced a faulty HDMI adapter and updated the display firmware."),
    ("Set up monitoring alerts for disk space", "Cloud & Servers", "Alerts configured at 80% and 90% disk usage with email notification."),
    ("Intermittent packet loss at branch office", "Network", "Faulty switch port and patch cable replaced."),
    ("Update employee offboarding checklist", "Other", "Checklist updated to include license reclaim and device return."),
]
SEED_ARTICLES = [
    ("Fix: VPN keeps disconnecting", "Network", "If remote staff lose the VPN every few minutes, check the firewall idle timeout, update the VPN client, and test on another network to rule out the home router."),
    ("Printer scan-to-email stopped working", "Hardware", "After a mail migration, update the printer SMTP server, port and credentials to the new relay, then send a test scan."),
    ("Backup job failing", "Cloud & Servers", "Check free space on the backup target, update the backup agent, and read the job log for the first error. Rerun the backup manually after fixing."),
    ("Reset a password or unlock an account", "Accounts & Access", "Use the admin console to reset the password, require a change at next sign-in, and unlock the account. If the reset portal shows an error, check that its certificate has not expired."),
    ("Outlook repeatedly asks for a password", "Email", "Clear saved credentials in Credential Manager, confirm modern authentication is on, and recreate the Outlook profile if it continues."),
    ("Shared mailbox not receiving external email", "Email", "Allow external senders on the mailbox, check mail flow rules, and look at the message trace for rejected messages."),
    ("Suspected ransomware or malware alert", "Security", "Disconnect the machine from the network, do not shut it down, run a full scan, check for cracked software, and alert the manager if files are encrypted."),
    ("Guest Wi-Fi setup checklist", "Network", "Use a separate VLAN, allow internet access only, block internal subnets, and rotate the password regularly."),
    ("Slow application or portal", "Software", "Check server CPU and memory, review slow database queries, and restart the application pool after changes."),
    ("Disk space alerts and cleanup", "Cloud & Servers", "Set alerts at 80% and 90% usage, clear old logs and backups, and resize volumes before they fill up."),
]


def seed():
    with Session(engine) as db:
        if not db.scalars(select(User).limit(1)).first():
            db.add(User(username="agent", name="Alex Rivera", role="agent", password_hash=hash_password("agent123")))
            db.add(User(username="manager", name="Morgan Lee", role="manager", password_hash=hash_password("manager123")))
        if not db.scalars(select(Article).limit(1)).first():
            for title, cat, content in SEED_ARTICLES:
                db.add(Article(title=title, category=cat, content=content))
        if not db.scalars(select(Ticket).limit(1)).first():
            rng = random.Random(42)
            companies = ["Meridian Dental Clinic", "Cascade Property Management",
                         "Summit Accounting Partners", "Greenfield Legal Group", "Harbor Logistics"]
            now = datetime.utcnow()
            for subject, cat, resolution in SEED_TICKETS:
                created = now - timedelta(days=rng.randint(0, 60), hours=rng.randint(0, 23))
                status = rng.choices(STATUSES, weights=[3, 3, 4])[0]
                t = Ticket(subject=subject, description="", company=rng.choice(companies),
                           priority=rng.choices(PRIORITIES, weights=[3, 4, 3])[0], status=status,
                           assignee=rng.choice(AGENTS), category=cat, created_at=created,
                           resolution=resolution if status == "Done" else "")
                if status == "Done":
                    t.resolved_at = min(now, created + timedelta(hours=rng.randint(2, 72)))
                db.add(t)
                db.flush()
                db.add(Activity(ticket_id=t.id, actor="System", action="Ticket created", created_at=created))
                if t.resolved_at:
                    db.add(Activity(ticket_id=t.id, actor=t.assignee, action="Marked as Done", created_at=t.resolved_at))
        db.commit()


# ---------- app ----------
app = FastAPI(title="Enterprise Operations & AI Support Platform")
app.add_middleware(SessionMiddleware, secret_key=os.getenv("SECRET_KEY", "dev-secret-change-me"),
                   same_site="lax", max_age=60 * 60 * 12)


@app.on_event("startup")
def startup():
    Base.metadata.create_all(engine)
    seed()


class LoginIn(BaseModel):
    username: str
    password: str


class TicketIn(BaseModel):
    subject: str
    description: str = ""
    company: str
    priority: str = "Medium"
    status: str = "To Do"
    assignee: str = "Unassigned"


class TicketPatch(BaseModel):
    subject: Optional[str] = None
    description: Optional[str] = None
    company: Optional[str] = None
    priority: Optional[str] = None
    status: Optional[str] = None
    assignee: Optional[str] = None
    resolution: Optional[str] = None


class ArticleIn(BaseModel):
    title: str
    category: str = "Other"
    content: str


def user_out(u: User):
    return {"username": u.username, "name": u.name, "role": u.role}


@app.post("/api/login")
def login(body: LoginIn, request: Request, db: Session = Depends(get_db)):
    u = db.scalars(select(User).where(User.username == body.username.strip().lower())).first()
    if not u or not verify_password(body.password, u.password_hash):
        raise HTTPException(401, "Wrong username or password")
    request.session["uid"] = u.id
    return user_out(u)


@app.post("/api/logout", status_code=204)
def logout(request: Request):
    request.session.clear()


@app.get("/api/me")
def me(user: User = Depends(current_user)):
    return user_out(user)


@app.get("/api/tickets")
def list_tickets(q: str = "", status: str = "", priority: str = "", company: str = "",
                 db: Session = Depends(get_db), user: User = Depends(current_user)):
    run_automation(db)
    stmt = select(Ticket).order_by(Ticket.created_at.desc())
    if status:
        stmt = stmt.where(Ticket.status == status)
    if priority:
        stmt = stmt.where(Ticket.priority == priority)
    if company:
        stmt = stmt.where(Ticket.company == company)
    if q:
        stmt = stmt.where(Ticket.subject.ilike(f"%{q}%"))
    return [out(t) for t in db.scalars(stmt)]


@app.get("/api/summary")
def summary(db: Session = Depends(get_db), user: User = Depends(current_user)):
    run_automation(db)
    tickets = list(db.scalars(select(Ticket)))
    open_t = [t for t in tickets if t.status != "Done"]
    return {
        "total": len(tickets), "open": len(open_t),
        "high_open": len([t for t in open_t if t.priority == "High"]),
        "escalated": len([t for t in open_t if t.escalated_at]),
        "done": len(tickets) - len(open_t),
        "companies": sorted({t.company for t in tickets}),
        "ai_enabled": ai.enabled(), "agents": AGENTS, "escalation_hours": ESCALATION_HOURS,
    }


@app.post("/api/tickets", status_code=201)
def create_ticket(body: TicketIn, db: Session = Depends(get_db), user: User = Depends(current_user)):
    data = body.model_dump()
    auto = data["priority"] == "Auto"
    if auto:
        data["priority"] = "Medium"
    check(data["priority"], data["status"])
    if not body.subject.strip() or not body.company.strip():
        raise HTTPException(422, "Subject and company are required")
    t = Ticket(**data)
    db.add(t)
    db.flush()
    log(db, t.id, user.name, "Ticket created")
    db.commit()
    if ai.enabled():  # tickets still work if the AI fails
        try:
            apply_ai(db, t, ai.analyze(t.subject, t.description), user.name, set_priority=auto)
            db.commit()
        except Exception as e:
            db.rollback()
            print("AI analysis failed:", e)
    db.refresh(t)
    return out(t)


@app.get("/api/tickets/{ticket_id}")
def ticket_detail(ticket_id: int, db: Session = Depends(get_db), user: User = Depends(current_user)):
    t = db.get(Ticket, ticket_id)
    if not t:
        raise HTTPException(404, "Ticket not found")
    acts = db.scalars(select(Activity).where(Activity.ticket_id == t.id).order_by(Activity.created_at.desc()))
    arts, past = similar_for(db, t)
    return {
        "ticket": out(t),
        "timeline": [{"actor": a.actor, "action": a.action, "created_at": iso(a.created_at)} for a in acts],
        "articles": [{"id": a.id, "title": a.title, "category": a.category, "content": a.content} for a in arts],
        "past": [{"id": o.id, "subject": o.subject, "company": o.company, "resolution": o.resolution} for o in past],
    }


@app.patch("/api/tickets/{ticket_id}")
def update_ticket(ticket_id: int, body: TicketPatch, db: Session = Depends(get_db),
                  user: User = Depends(current_user)):
    t = db.get(Ticket, ticket_id)
    if not t:
        raise HTTPException(404, "Ticket not found")
    data = body.model_dump(exclude_none=True)
    check(data.get("priority"), data.get("status"))
    for k, v in data.items():
        old = getattr(t, k)
        if old == v:
            continue
        setattr(t, k, v)
        if k in ("status", "priority", "assignee"):
            log(db, t.id, user.name, f"{k.capitalize()} changed from {old} to {v}")
        elif k == "resolution":
            log(db, t.id, user.name, "Resolution note updated")
    if "status" in data:
        t.resolved_at = (t.resolved_at or datetime.utcnow()) if t.status == "Done" else None
    db.commit()
    db.refresh(t)
    return out(t)


@app.post("/api/tickets/{ticket_id}/analyze")
def analyze_ticket(ticket_id: int, db: Session = Depends(get_db), user: User = Depends(current_user)):
    t = db.get(Ticket, ticket_id)
    if not t:
        raise HTTPException(404, "Ticket not found")
    if not ai.enabled():
        raise HTTPException(503, "AI is not set up. Add GEMINI_API_KEY to the .env file and restart.")
    try:
        apply_ai(db, t, ai.analyze(t.subject, t.description), user.name, set_priority=True)
    except Exception as e:
        db.rollback()
        raise HTTPException(502, f"The AI request failed: {str(e)[:200]}")
    db.commit()
    db.refresh(t)
    return out(t)


@app.delete("/api/tickets/{ticket_id}", status_code=204)
def delete_ticket(ticket_id: int, db: Session = Depends(get_db), user: User = Depends(manager_only)):
    t = db.get(Ticket, ticket_id)
    if not t:
        raise HTTPException(404, "Ticket not found")
    log(db, ticket_id, user.name, f"Ticket deleted: {t.subject}")
    db.delete(t)
    db.commit()


# ----- knowledge base -----
@app.get("/api/articles")
def list_articles(q: str = "", db: Session = Depends(get_db), user: User = Depends(current_user)):
    stmt = select(Article).order_by(Article.id.desc())
    if q:
        like = f"%{q}%"
        stmt = stmt.where(Article.title.ilike(like) | Article.content.ilike(like))
    return [{"id": a.id, "title": a.title, "category": a.category, "content": a.content}
            for a in db.scalars(stmt)]


@app.post("/api/articles", status_code=201)
def create_article(body: ArticleIn, db: Session = Depends(get_db), user: User = Depends(current_user)):
    if not body.title.strip() or not body.content.strip():
        raise HTTPException(422, "Title and content are required")
    a = Article(title=body.title.strip(), category=body.category, content=body.content.strip())
    db.add(a)
    db.commit()
    return {"id": a.id}


@app.delete("/api/articles/{article_id}", status_code=204)
def delete_article(article_id: int, db: Session = Depends(get_db), user: User = Depends(manager_only)):
    a = db.get(Article, article_id)
    if not a:
        raise HTTPException(404, "Article not found")
    db.delete(a)
    db.commit()


# ----- automation, activity, analytics -----
@app.get("/api/automation")
def automation(db: Session = Depends(get_db), user: User = Depends(current_user)):
    run_automation(db)
    esc = db.scalars(select(Ticket).where(Ticket.escalated_at.is_not(None), Ticket.status != "Done")
                     .order_by(Ticket.escalated_at.desc()))
    return {"ai_enabled": ai.enabled(), "escalation_hours": ESCALATION_HOURS,
            "category_owner": CATEGORY_OWNER, "escalated": [out(t) for t in esc]}


@app.get("/api/activity")
def activity(db: Session = Depends(get_db), user: User = Depends(current_user)):
    rows = db.scalars(select(Activity).order_by(Activity.created_at.desc()).limit(100))
    result = []
    for a in rows:
        t = db.get(Ticket, a.ticket_id) if a.ticket_id else None
        result.append({"actor": a.actor, "action": a.action, "created_at": iso(a.created_at),
                       "ticket_id": a.ticket_id, "subject": t.subject if t else "(deleted ticket)"})
    return result


def compute_analytics(db: Session):
    run_automation(db)
    tickets = list(db.scalars(select(Ticket)))
    now = datetime.utcnow()
    resolved = [t for t in tickets if t.resolved_at]
    avg = (sum((t.resolved_at - t.created_at).total_seconds() for t in resolved) / len(resolved) / 3600
           if resolved else 0)
    weekly = []
    for i in range(7, -1, -1):
        end, start = now - timedelta(days=7 * i), now - timedelta(days=7 * (i + 1))
        weekly.append({"label": start.strftime("%d %b"),
                       "count": len([t for t in tickets if start <= t.created_at < end])})
    recent = [t for t in tickets if t.created_at >= now - timedelta(days=30)]
    repeat = [{"category": c, "count": n} for c, n in Counter(t.category for t in recent).most_common(5) if n >= 2]
    return {
        "total": len(tickets), "open": len([t for t in tickets if t.status != "Done"]),
        "escalated_open": len([t for t in tickets if t.escalated_at and t.status != "Done"]),
        "avg_resolution_hours": round(avg, 1),
        "by_category": dict(Counter(t.category for t in tickets).most_common()),
        "by_priority": {p: len([t for t in tickets if t.priority == p]) for p in PRIORITIES},
        "by_status": {s: len([t for t in tickets if t.status == s]) for s in STATUSES},
        "by_company": dict(Counter(t.company for t in tickets).most_common(5)),
        "weekly": weekly, "repeat_issues": repeat,
    }


@app.get("/api/analytics")
def analytics(db: Session = Depends(get_db), user: User = Depends(manager_only)):
    return compute_analytics(db)


@app.post("/api/report")
def report(db: Session = Depends(get_db), user: User = Depends(manager_only)):
    a = compute_analytics(db)
    text = (
        f"Total tickets: {a['total']}. Open: {a['open']}. Escalated and still open: {a['escalated_open']}. "
        f"Average resolution time: {a['avg_resolution_hours']} hours.\n"
        f"Tickets by category: {a['by_category']}.\n"
        f"Tickets by priority: {a['by_priority']}. By status: {a['by_status']}.\n"
        f"New tickets per week, oldest to newest: {[w['count'] for w in a['weekly']]}.\n"
        f"Categories with repeat issues in the last 30 days: {a['repeat_issues']}."
    )
    if ai.enabled():
        try:
            return {"report": ai.weekly_report(text), "source": "ai"}
        except Exception as e:
            print("AI report failed:", e)
    return {"report": text, "source": "summary"}


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def index():
    return FileResponse("static/index.html")
