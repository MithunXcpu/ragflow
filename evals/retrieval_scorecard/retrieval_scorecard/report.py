"""Static HTML scorecard."""

from __future__ import annotations

import html
from datetime import datetime, timezone

from retrieval_scorecard.models import ConfigScore


def render_html(
    config_scores: list[ConfigScore],
    *,
    mode: str,
    embedding_backend: str,
    generated_at: datetime | None = None,
    notes: list[str] | None = None,
) -> str:
    stamp = (generated_at or datetime.now(timezone.utc)).strftime("%Y-%m-%d %H:%M UTC")
    notes = notes or []
    rows = "\n".join(_summary_row(item) for item in config_scores)
    detail = "\n".join(_detail_section(item) for item in config_scores)
    note_html = "".join(f"<li>{html.escape(note)}</li>" for note in notes)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Retrieval Evals Scorecard</title>
<style>
  :root {{
    --ink: #1c1915;
    --muted: #5c564c;
    --line: #d9d1c4;
    --paper: #f6f3ec;
    --card: #fffdf8;
    --good: #0f6e56;
    --mid: #9a6700;
    --low: #9f2d1c;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    font: 15px/1.5 "Iowan Old Style", Palatino, "Palatino Linotype", Georgia, serif;
    color: var(--ink);
    background: var(--paper);
  }}
  main {{ max-width: 1080px; margin: 0 auto; padding: 32px 20px 64px; }}
  h1 {{ font-size: 32px; line-height: 1.15; margin: 0 0 8px; }}
  h2 {{ font-size: 20px; margin: 32px 0 8px; }}
  p {{ margin: 8px 0; }}
  .banner {{
    background: #efe4c4;
    border: 1px solid #d8c48a;
    padding: 10px 12px;
    margin: 16px 0 20px;
  }}
  .meta {{ color: var(--muted); font-family: ui-sans-serif, sans-serif; font-size: 13px; }}
  table {{
    width: 100%;
    border-collapse: collapse;
    background: var(--card);
    font-family: ui-sans-serif, sans-serif;
    font-size: 13px;
  }}
  th, td {{ border-bottom: 1px solid var(--line); text-align: left; padding: 8px 10px; vertical-align: top; }}
  th {{ font-size: 12px; letter-spacing: 0.02em; color: var(--muted); }}
  td.num, th.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  .bar {{
    height: 8px;
    background: #ece6db;
    margin-top: 6px;
  }}
  .bar > span {{ display: block; height: 8px; background: var(--good); }}
  .miss {{ color: var(--low); }}
  .hit {{ color: var(--good); }}
  code {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }}
</style>
</head>
<body>
<main>
  <h1>Retrieval Evals Scorecard</h1>
  <p class="meta">Generated {html.escape(stamp)} · mode <code>{html.escape(mode)}</code> · embeddings <code>{html.escape(embedding_backend)}</code></p>
  <div class="banner">
    SAMPLE DATA. The Harborline field-service pages and the 20 questions are fictional demo material for this scorecard. They are not a customer result, a benchmark claim, or a production measurement.
  </div>
  <p>
    Each row is the same golden set run under a different retrieval configuration.
    Recall@k is the fraction of expected source chunks that appear in the top k hits.
    MRR is the mean of 1/rank for the first expected chunk.
    Faithfulness asks whether the drafted answer stays inside the retrieved text.
    The offline judge drafts from the retrieved sentences. If those sentences do not contain the reference values, it appends the reference answer and counts the missing values as unsupported. That is a stand-in for a model that answers from memory when retrieval misses. It is not a claim about a hosted judge.
  </p>
  <h2>Comparison</h2>
  <table>
    <thead>
      <tr>
        <th>Config</th>
        <th>Chunk window</th>
        <th class="num">Top-k</th>
        <th>Rerank</th>
        <th class="num">Recall@k</th>
        <th class="num">MRR</th>
        <th class="num">Faithfulness</th>
        <th>Judge</th>
      </tr>
    </thead>
    <tbody>
      {rows}
    </tbody>
  </table>
  {detail}
  <h2>Run notes</h2>
  <ul>
    {note_html}
  </ul>
</main>
</body>
</html>
"""


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _bar(value: float) -> str:
    width = max(0.0, min(value, 1.0)) * 100
    return f'<div class="bar"><span style="width:{width:.1f}%"></span></div>'


def _summary_row(item: ConfigScore) -> str:
    config = item.config
    judge = item.questions[0].judge if item.questions else ""
    return (
        "<tr>"
        f"<td><strong>{html.escape(config.name)}</strong><br>{html.escape(config.description)}</td>"
        f"<td>{config.chunk_chars} chars</td>"
        f'<td class="num">{config.top_k}</td>'
        f"<td>{'on' if config.rerank else 'off'}</td>"
        f'<td class="num">{_pct(item.recall_at_k)}{_bar(item.recall_at_k)}</td>'
        f'<td class="num">{item.mrr:.3f}{_bar(item.mrr)}</td>'
        f'<td class="num">{item.faithfulness:.3f}{_bar(item.faithfulness)}</td>'
        f"<td><code>{html.escape(judge)}</code></td>"
        "</tr>"
    )


def _detail_section(item: ConfigScore) -> str:
    rows = []
    for question in item.questions:
        if question.first_relevant_rank is None:
            rank = '<span class="miss">miss</span>'
        else:
            rank = f'<span class="hit">{question.first_relevant_rank}</span>'
        rows.append(
            "<tr>"
            f"<td><code>{html.escape(question.question_id)}</code></td>"
            f"<td>{html.escape(question.question)}</td>"
            f"<td><code>{html.escape(', '.join(question.expected_chunks))}</code></td>"
            f'<td class="num">{rank}</td>'
            f'<td class="num">{question.recall_at_k:.2f}</td>'
            f'<td class="num">{question.faithfulness:.2f}</td>'
            "</tr>"
        )
    body = "\n".join(rows)
    return f"""
  <h2>{html.escape(item.config.name)}</h2>
  <p class="meta">{html.escape(item.backend)}</p>
  <table>
    <thead>
      <tr>
        <th>ID</th>
        <th>Question</th>
        <th>Expected chunk</th>
        <th class="num">First hit</th>
        <th class="num">Recall</th>
        <th class="num">Faithfulness</th>
      </tr>
    </thead>
    <tbody>
      {body}
    </tbody>
  </table>
"""
