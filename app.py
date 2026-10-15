"""Exercise 5 — Streamlit approval UI for the HITL PR review agent.

Run with:
    uv run streamlit run app.py

Goal: wrap the LangGraph built in exercises 1–4 in a web UI that adapts to
the confidence bucket of each PR.

Routing thresholds (common/schemas.py):
    > 72%        auto_approve     UI shows a success card; reviewer does nothing
    58 – 72%     human_approval   UI shows Approve / Reject / Edit buttons
    <  58%       escalate         UI shows a question form for the reviewer
"""

from __future__ import annotations

import asyncio
import uuid

import streamlit as st
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from common.db import db_conn, db_path, replay_events
from exercises.exercise_4_audit import build_graph

load_dotenv()


# ─── Session state ─────────────────────────────────────────────────────────
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "pr_url" not in st.session_state:
    st.session_state.pr_url = ""
if "interrupt_payload" not in st.session_state:
    st.session_state.interrupt_payload = None
if "final" not in st.session_state:
    st.session_state.final = None


# ─── Page setup ────────────────────────────────────────────────────────────
st.set_page_config(page_title="HITL PR Review", layout="wide")
st.title("HITL PR Review Agent")


# ─── Database helper ───────────────────────────────────────────────────────
async def fetch_recent_sessions():
    try:
        async with db_conn() as conn:
            async with conn.execute(
                """
                SELECT thread_id,
                       pr_url,
                       MIN(timestamp)        AS started,
                       MAX(timestamp)        AS last_event,
                       MAX(risk_level)       AS worst_risk,
                       COUNT(*)              AS events
                  FROM audit_events
                 GROUP BY thread_id, pr_url
                 ORDER BY MAX(timestamp) DESC
                 LIMIT 25
                """
            ) as cur:
                rows = await cur.fetchall()
                return [dict(r) for r in rows]
    except Exception:
        return []


# ─── Sidebar — recent sessions ─────────────────────────────────────────────
with st.sidebar:
    st.header("Recent sessions")
    sessions = asyncio.run(fetch_recent_sessions())
    if not sessions:
        st.caption("No recent sessions found.")
    else:
        for s in sessions:
            t_id = s["thread_id"]
            short_id = t_id[:8] + "..."
            pr_tail = s["pr_url"].split("/")[-1] if "/" in s["pr_url"] else s["pr_url"]
            risk = s.get("worst_risk", "unknown")
            btn_label = f"PR #{pr_tail} [{risk}] — {short_id}"
            if st.button(btn_label, key=f"session_btn_{t_id}", use_container_width=True):
                st.session_state.thread_id = t_id
                st.session_state.pr_url = s["pr_url"]
                st.session_state.interrupt_payload = None
                st.session_state.final = None
                st.rerun()

        if st.session_state.thread_id:
            st.divider()
            st.markdown(f"**Current thread:**\n`{st.session_state.thread_id}`")
            with st.expander("Session timeline", expanded=False):
                try:
                    events = asyncio.run(replay_events(st.session_state.thread_id))
                    for ev in events:
                        rev = f" ({ev['reviewer_id']})" if ev.get("reviewer_id") else ""
                        st.markdown(f"- **{ev['action']}**: `{ev['decision']}` | conf `{ev['confidence']:.0%}`{rev}")
                except Exception as e:
                    st.caption(f"Error loading timeline: {e}")


# ─── Top form — start a new review ─────────────────────────────────────────
with st.form("start"):
    pr_url = st.text_input(
        "PR URL", value=st.session_state.pr_url,
        placeholder="https://github.com/VinUni-AI20k/PR-Demo/pull/1",
    )
    submitted = st.form_submit_button("Run review", type="primary")


# ─── Renderers per interrupt kind ──────────────────────────────────────────
def render_approval_card(payload: dict) -> dict | None:
    """58–72% bucket: show the LLM review + 3 buttons. Return resume dict or None."""
    conf = payload["confidence"]
    st.subheader(f"Approval requested — confidence {conf:.0%}")
    st.caption(payload["confidence_reasoning"])
    st.markdown(payload["summary"])

    for c in payload.get("comments", []):
        st.markdown(f"- **[{c['severity']}]** `{c['file']}:{c.get('line') or '?'}` — {c['body']}")

    with st.expander("Diff"):
        st.code(payload.get("diff_preview", ""), language="diff")

    feedback = st.text_input("Feedback (optional)", key="approval_feedback")
    col1, col2, col3 = st.columns(3)
    if col1.button("Approve", type="primary", use_container_width=True):
        return {"choice": "approve", "feedback": feedback}
    if col2.button("Reject", use_container_width=True):
        return {"choice": "reject", "feedback": feedback}
    if col3.button("Edit", use_container_width=True):
        return {"choice": "edit", "feedback": feedback}
    return None


def render_escalation_card(payload: dict) -> dict | None:
    """< 58% bucket: show risk factors + question form. Return {question: answer} or None."""
    conf = payload["confidence"]
    st.subheader(f"Strong escalation — confidence {conf:.0%}")
    if payload.get("confidence_reasoning"):
        st.caption(payload["confidence_reasoning"])
    if payload.get("risk_factors"):
        st.error("Risks: " + ", ".join(payload["risk_factors"]))
    st.markdown(payload["summary"])

    with st.form("escalation"):
        st.markdown("### Escalation Questions")
        answers: dict[str, str] = {}
        for idx, q in enumerate(payload.get("questions", [])):
            answers[q] = st.text_input(q, key=f"escalation_q_{idx}")
        submitted_answers = st.form_submit_button("Submit answers", type="primary")
        if submitted_answers:
            return answers
    return None


# ─── Drive the graph ───────────────────────────────────────────────────────
async def run_graph(pr_url: str, thread_id: str, resume_value=None):
    """Invoke the graph once. Returns the final result or {'__interrupt__': ...}."""
    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        app = build_graph(cp)
        cfg = {"configurable": {"thread_id": thread_id}}

        if resume_value is None:
            result = await app.ainvoke({"pr_url": pr_url, "thread_id": thread_id}, cfg)
        else:
            result = await app.ainvoke(Command(resume=resume_value), cfg)
        return result


# ─── Main flow ─────────────────────────────────────────────────────────────
if submitted and pr_url:
    st.session_state.pr_url = pr_url
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.interrupt_payload = None
    st.session_state.final = None

    with st.spinner("Fetching PR + asking the LLM..."):
        try:
            result = asyncio.run(run_graph(pr_url, st.session_state.thread_id))
            if "__interrupt__" in result:
                st.session_state.interrupt_payload = result["__interrupt__"][0].value
            else:
                st.session_state.final = result
        except Exception as e:
            st.error(f"Error running review: {e}")

# Render the current interrupt card, if any
payload = st.session_state.interrupt_payload
if payload is not None:
    kind = payload["kind"]
    answer = render_approval_card(payload) if kind == "approval_request" else render_escalation_card(payload)
    if answer is not None:
        with st.spinner("Resuming..."):
            try:
                result = asyncio.run(run_graph(
                    st.session_state.pr_url, st.session_state.thread_id, resume_value=answer,
                ))
                if "__interrupt__" in result:
                    st.session_state.interrupt_payload = result["__interrupt__"][0].value
                else:
                    st.session_state.interrupt_payload = None
                    st.session_state.final = result
                st.rerun()
            except Exception as e:
                st.error(f"Error resuming review: {e}")

# Render final state, if reached
if st.session_state.final is not None:
    final = st.session_state.final
    action = final.get("final_action", "?")
    analysis = final.get("analysis")
    if action.startswith("auto") or action.startswith("committed"):
        st.success(f"✓ {action} — comment posted to {st.session_state.pr_url}")
        if action.startswith("auto") and analysis:
            st.info(f"**Confidence**: {analysis.confidence:.0%}\n\n**Summary**: {analysis.summary}")
        if st.session_state.pr_url:
            st.markdown(f"[🔗 View Pull Request on GitHub]({st.session_state.pr_url})")
    elif action == "rejected":
        st.warning("Rejected — no comment posted")
        if analysis:
            st.info(f"**Confidence**: {analysis.confidence:.0%}\n\n**Summary**: {analysis.summary}")
    else:
        st.info(f"final_action = {action}")
    st.caption(f"thread_id = `{st.session_state.thread_id}`  ·  replay: "
               f"`uv run python -m audit.replay --thread {st.session_state.thread_id}`")
