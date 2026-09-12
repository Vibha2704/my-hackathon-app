"""
app.py
Streamlit UI for the Smart Shortlisting Engine.

Run this with:
    streamlit run app.py
(NOT "python app.py" -- Streamlit apps are launched differently.)
"""

import os
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from matcher import extract_text, rank_resumes, explain_candidate

TEMP_DIR = "temp_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Shortlist — Smart Shortlisting Engine", page_icon="◆", layout="wide")

INK = "#15171A"
INK_SOFT = "#5B5F66"
BORDER = "#E1E0DB"
SURFACE = "#F6F6F4"
ACCENT = "#1B4332"
ACCENT_SOFT = "#E4EDE8"
WARN = "#9C6B54"
WARN_SOFT = "#F5EDE7"
LILAC = "#7E6BA6"
LILAC_DARK = "#63517F"
LILAC_SOFT = "#F1EEF8"


def render_html(html: str) -> None:
    """Render raw HTML via st.markdown.

    Streamlit's markdown parser treats any line indented 4+ spaces as a
    preformatted code block. Because these HTML strings are built inside
    nested functions/loops, their natural Python indentation can trip that
    rule and print literal tags instead of rendering them. Stripping each
    line's leading whitespace first avoids that while leaving the HTML's
    own meaning untouched.
    """
    lines = [ln.lstrip() for ln in html.strip("\n").split("\n")]
    st.markdown("\n".join(lines), unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Global styling
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Source+Serif+4:opsz,wght@8..60,500;8..60,600&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, sans-serif;
        color: {INK};
    }}

    #MainMenu, header[data-testid="stHeader"] {{ background: transparent; }}
    .stApp {{ background: #FFFFFF; }}
    .block-container {{ padding-top: 2.2rem; max-width: 1180px; }}

    /* ---------- Masthead ---------- */
    .masthead {{
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        border-bottom: 1px solid {BORDER};
        padding-bottom: 18px;
        margin-bottom: 6px;
    }}
    .masthead h1 {{
        font-family: 'Source Serif 4', serif;
        font-weight: 600;
        font-size: 2.05rem;
        letter-spacing: -0.01em;
        margin: 0;
        color: {INK};
    }}
    .masthead .tag {{
        color: {INK_SOFT};
        font-size: 0.92rem;
        max-width: 360px;
        text-align: right;
        line-height: 1.4;
    }}

    /* ---------- Stat strip (mirrors landing-page metric row) ---------- */
    .stat-strip {{
        display: flex;
        gap: 0;
        margin: 28px 0 8px 0;
    }}
    .stat {{
        flex: 1;
        padding: 0 22px;
        border-left: 1px solid {BORDER};
    }}
    .stat:first-child {{ border-left: none; padding-left: 0; }}
    .stat .num {{
        font-family: 'Source Serif 4', serif;
        font-size: 2.1rem;
        font-weight: 600;
        line-height: 1;
        color: {INK};
    }}
    .stat .label {{
        color: {INK_SOFT};
        font-size: 0.82rem;
        margin-top: 6px;
    }}

    /* ---------- Upload zone ---------- */
    div[data-testid="stFileUploaderDropzone"] {{
        background: {SURFACE};
        border: 1px dashed {BORDER};
        border-radius: 6px;
    }}
    div[data-testid="stFileUploaderDropzone"]:hover {{
        border-color: {LILAC};
        transition: border-color 180ms ease;
    }}

    /* ---------- Buttons ---------- */
    .stButton > button {{
        background: {LILAC};
        color: #FFFFFF;
        border: none;
        border-radius: 8px;
        padding: 0.55rem 1.4rem;
        font-weight: 500;
        transition: transform 120ms ease, background 120ms ease;
    }}
    .stButton > button:hover {{
        background: {LILAC_DARK};
        transform: translateY(-1px);
        color: #FFFFFF;
    }}

    /* ---------- Tabs, minimal underline style ---------- */
    button[data-baseweb="tab"] {{
        font-weight: 500;
        color: {INK_SOFT};
    }}
    button[data-baseweb="tab"][aria-selected="true"] {{
        color: {INK};
    }}
    div[data-baseweb="tab-highlight"] {{
        background-color: {LILAC} !important;
        height: 2px !important;
    }}

    /* ---------- Sidebar sliders (native widgets, tinted to match) --- */
    [data-testid="stSlider"] [role="slider"] {{
        background-color: {LILAC} !important;
        border-color: {LILAC} !important;
    }}
    [data-testid="stSlider"] div[data-baseweb="slider"] > div > div {{
        background: {LILAC} !important;
    }}

    /* ---------- Chips ---------- */
    .chip {{
        display: inline-block;
        padding: 3px 11px;
        margin: 3px 6px 3px 0;
        border-radius: 20px;
        font-size: 0.82rem;
        line-height: 1.6;
    }}
    .chip-neutral {{ background: {SURFACE}; color: {INK_SOFT}; border: 1px solid {BORDER}; }}
    .chip-match {{ background: {ACCENT_SOFT}; color: {ACCENT}; }}
    .chip-gap {{ background: #FFFFFF; color: {WARN}; border: 1px solid {WARN}; }}

    /* ---------- Ranking table ---------- */
    table.rank-table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 0.9rem;
    }}
    table.rank-table th {{
        text-align: left;
        color: {INK_SOFT};
        font-weight: 500;
        font-size: 0.78rem;
        padding: 8px 10px;
        border-bottom: 1px solid {BORDER};
    }}
    table.rank-table td {{
        padding: 10px 10px;
        border-bottom: 1px solid {BORDER};
        vertical-align: middle;
    }}
    table.rank-table tr:hover td {{
        background: {LILAC_SOFT};
        transition: background 150ms ease;
    }}
    .bar-track {{
        width: 100%;
        height: 6px;
        background: {SURFACE};
        border-radius: 3px;
        overflow: hidden;
    }}
    .bar-fill {{
        height: 100%;
        border-radius: 3px;
    }}
    .rank-num {{
        font-family: 'Source Serif 4', serif;
        color: {INK_SOFT};
        font-size: 0.95rem;
    }}
    .rank-num.top {{ color: {ACCENT}; font-weight: 600; }}

    /* ---------- Candidate cards ---------- */
    .candidate-card {{
        border: 1px solid {BORDER};
        border-radius: 6px;
        padding: 18px 20px;
        margin-bottom: 14px;
        background: #FFFFFF;
    }}
    .candidate-card .head {{
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        margin-bottom: 10px;
    }}
    .candidate-card .name {{
        font-weight: 600;
        font-size: 1.02rem;
    }}
    .candidate-card .score {{
        font-family: 'Source Serif 4', serif;
        font-size: 1.3rem;
        color: {ACCENT};
    }}
    .composition {{
        display: flex;
        height: 8px;
        border-radius: 4px;
        overflow: hidden;
        margin: 10px 0 8px 0;
    }}
    .legend-dot {{
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 5px;
    }}
    .legend {{ color: {INK_SOFT}; font-size: 0.78rem; margin-bottom: 4px; }}

    details.explain summary {{
        cursor: pointer;
        color: {INK_SOFT};
        font-size: 0.85rem;
        margin-top: 4px;
        outline: none;
    }}
    details.explain summary:hover {{ color: {INK}; }}
    details.explain[open] summary {{ color: {ACCENT}; }}
    details.explain .body {{
        padding-top: 10px;
        font-size: 0.88rem;
        color: {INK};
        line-height: 1.65;
        animation: reveal 180ms ease;
    }}

    @keyframes reveal {{
        from {{ opacity: 0; transform: translateY(-4px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    @keyframes fadeUp {{
        from {{ opacity: 0; transform: translateY(10px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    .fade-1 {{ animation: fadeUp 420ms ease both; }}
    .fade-2 {{ animation: fadeUp 420ms ease 90ms both; }}
    .fade-3 {{ animation: fadeUp 420ms ease 180ms both; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Masthead
# ---------------------------------------------------------------------------
render_html(
    """
    <div class="masthead">
        <h1>Shortlist</h1>
        <div class="tag">Upload a job description and a batch of resumes to get a ranked, explainable shortlist.</div>
    </div>
    """
)

# ---------------------------------------------------------------------------
# Sidebar — scoring configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("**Scoring weights**")
    st.caption("Adjust how much each signal contributes to the final score.")
    keyword_weight = st.slider("Skill keywords", 0.0, 1.0, 0.4, 0.05)
    semantic_weight = st.slider("Semantic fit", 0.0, 1.0, 0.4, 0.05)
    criteria_weight = st.slider("Education / experience / other", 0.0, 1.0, 0.2, 0.05)

    total_w = keyword_weight + semantic_weight + criteria_weight
    if total_w == 0:
        keyword_weight, semantic_weight, criteria_weight = 0.4, 0.4, 0.2
        total_w = 1.0
    keyword_weight, semantic_weight, criteria_weight = (
        keyword_weight / total_w,
        semantic_weight / total_w,
        criteria_weight / total_w,
    )

    st.divider()
    with st.expander("How the scoring works"):
        st.markdown(
            """
            - **Skill keywords** — does the resume mention the tools and terms explicitly called for in the JD (with fuzzy matching for typos and variants)?
            - **Semantic fit** — a latent-semantic-analysis comparison of overall meaning, so relevant experience described in different words still counts.
            - **Other criteria** — education, years of experience, CGPA, location, and certifications, only for what the JD actually specifies.
            """
        )


def save_upload(uploaded_file):
    path = os.path.join(TEMP_DIR, uploaded_file.name)
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path


# ---------------------------------------------------------------------------
# Upload zone
# ---------------------------------------------------------------------------
col1, col2 = st.columns(2)
with col1:
    jd_files = st.file_uploader(
        "Job description(s)",
        type=["pdf", "docx", "html", "htm", "txt"],
        accept_multiple_files=True,
        help="Upload one JD to rank against a single role, or several to compare candidates across multiple roles.",
    )
with col2:
    resume_files = st.file_uploader(
        "Resumes",
        type=["pdf", "docx", "html", "htm", "txt"],
        accept_multiple_files=True,
        help="Select every resume at once.",
    )

run_button = st.button("Rank candidates", type="primary")

# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def chip_row(items, kind):
    if not items:
        return f'<span style="color:{INK_SOFT}; font-size:0.85rem;">None</span>'
    return "".join(f'<span class="chip chip-{kind}">{i}</span>' for i in items)


def score_bar_chart(results):
    names = [r["filename"] for r in results][::-1]
    scores = [r["final_score"] for r in results][::-1]
    colors = [ACCENT if i == len(results) - 1 else "#C9C8C3" for i in range(len(results))][::-1]

    fig = go.Figure(
        go.Bar(
            x=scores,
            y=names,
            orientation="h",
            marker=dict(color=colors),
            hovertemplate="%{y}<br>Score: %{x:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=max(220, 34 * len(names)),
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color=INK, size=13),
        xaxis=dict(range=[0, 1], showgrid=False, zeroline=False, title=None),
        yaxis=dict(showgrid=False, zeroline=False),
        bargap=0.35,
    )
    return fig


def render_ranking_table(results):
    rows = []
    for i, r in enumerate(results, start=1):
        pct = int(round(r["final_score"] * 100))
        bar_color = ACCENT if i == 1 else INK_SOFT
        rank_cls = "rank-num top" if i == 1 else "rank-num"
        # Built as one line on purpose -- an indented multi-line f-string here
        # gets misread by Markdown as a code block instead of rendered HTML.
        row = (
            f'<tr><td class="{rank_cls}">{i:02d}</td>'
            f'<td>{r["filename"]}</td>'
            f'<td style="width: 30%;"><div class="bar-track">'
            f'<div class="bar-fill" style="width:{pct}%; background:{bar_color};"></div></div></td>'
            f'<td style="text-align:right;">{r["final_score"]:.2f}</td>'
            f'<td style="text-align:right; color:{INK_SOFT};">{r["keyword_score"]:.2f}</td>'
            f'<td style="text-align:right; color:{INK_SOFT};">{r["semantic_score"]:.2f}</td></tr>'
        )
        rows.append(row)

    table_html = (
        '<table class="rank-table"><thead><tr>'
        "<th>Rank</th><th>Candidate</th><th>Fit</th>"
        '<th style="text-align:right;">Final</th>'
        '<th style="text-align:right;">Keywords</th>'
        '<th style="text-align:right;">Semantic</th>'
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )
    render_html(table_html)


def render_candidate_card(rank, r, required_skills, keyword_weight, semantic_weight, criteria_weight):
    kw_part = keyword_weight * r["keyword_score"]
    sem_part = semantic_weight * r["semantic_score"]
    crit_part = criteria_weight * r["criteria_score"]
    total = max(kw_part + sem_part + crit_part, 1e-6)

    seg_html = (
        f'<div style="width:{kw_part/total*100:.1f}%; background:{LILAC};"></div>'
        f'<div style="width:{sem_part/total*100:.1f}%; background:#B9B4C7;"></div>'
        f'<div style="width:{crit_part/total*100:.1f}%; background:{ACCENT};"></div>'
    )

    explanation = explain_candidate(r, required_skills).replace("**", "")

    card_html = f"""
    <div class="candidate-card">
        <div class="head">
            <span class="name">#{rank} — {r['filename']}</span>
            <span class="score">{r['final_score']:.2f}</span>
        </div>
        <div class="composition">{seg_html}</div>
        <div class="legend">
            <span class="legend-dot" style="background:{LILAC};"></span>Keywords
            &nbsp;&nbsp;<span class="legend-dot" style="background:#B9B4C7;"></span>Semantic
            &nbsp;&nbsp;<span class="legend-dot" style="background:{ACCENT};"></span>Other criteria
        </div>
        <div>{chip_row(r['found'], 'match')}{chip_row(r['missing'], 'gap')}</div>
        <details class="explain">
            <summary>View full breakdown</summary>
            <div class="body">{explanation.replace(chr(10), '<br>')}</div>
        </details>
    </div>
    """
    render_html(card_html)


def render_results(jd_name, results, required_skills, weights):
    keyword_weight, semantic_weight, criteria_weight = weights

    render_html(
        f"<div class='legend' style='margin-top:6px;'>Skills detected in JD</div>{chip_row(required_skills, 'neutral')}"
    )

    left, right = st.columns([3, 2])
    with left:
        render_html("<div class='fade-1'>")
        st.plotly_chart(score_bar_chart(results), use_container_width=True, config={"displayModeBar": False})
        render_html("</div>")
    with right:
        render_html("<div class='fade-2'>")
        render_ranking_table(results)
        render_html("</div>")

    render_html("<div class='fade-3'>")
    st.markdown("#### Top matches")
    for i, r in enumerate(results[:3], start=1):
        render_candidate_card(i, r, required_skills, keyword_weight, semantic_weight, criteria_weight)
    render_html("</div>")


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------
if run_button:
    if not jd_files or not resume_files:
        st.warning("Upload at least one job description and at least one resume to continue.")
    else:
        with st.spinner("Reading files and computing matches..."):
            resumes = {}
            for rf in resume_files:
                path = save_upload(rf)
                resumes[rf.name] = extract_text(path)

            jd_results = {}
            for jd_file in jd_files:
                jd_path = save_upload(jd_file)
                jd_text = extract_text(jd_path)
                results, required_skills = rank_resumes(
                    jd_text,
                    resumes,
                    keyword_weight=keyword_weight,
                    semantic_weight=semantic_weight,
                    criteria_weight=criteria_weight,
                )
                jd_results[jd_file.name] = (results, required_skills)

        st.toast(f"Ranked {len(resumes)} candidates against {len(jd_files)} job description(s).", icon="✅")

        top_overall = max(
            (results[0]["final_score"] for results, _ in jd_results.values()), default=0.0
        )
        render_html(
            f"""
            <div class="stat-strip fade-1">
                <div class="stat"><div class="num">{len(resumes)}</div><div class="label">Candidates ranked</div></div>
                <div class="stat"><div class="num">{len(jd_files)}</div><div class="label">Job description{'s' if len(jd_files) != 1 else ''}</div></div>
                <div class="stat"><div class="num">{top_overall:.2f}</div><div class="label">Top score</div></div>
            </div>
            """
        )

        weights = (keyword_weight, semantic_weight, criteria_weight)

        if len(jd_results) > 1:
            tabs = st.tabs(list(jd_results.keys()))
            for tab, (jd_name, (results, required_skills)) in zip(tabs, jd_results.items()):
                with tab:
                    render_results(jd_name, results, required_skills, weights)
        else:
            jd_name, (results, required_skills) = list(jd_results.items())[0]
            render_results(jd_name, results, required_skills, weights)
else:
    render_html(
        f"""
        <div style="border:1px solid {BORDER}; border-radius:6px; padding:28px 24px; margin-top:18px; color:{INK_SOFT}; font-size:0.92rem;">
            Upload files above, then run <b style="color:{INK};">Rank candidates</b> to see a scored, explainable shortlist —
            skill coverage, semantic fit, and any education, experience, CGPA, location, or certification requirements the JD specifies.
        </div>
        """
    )
