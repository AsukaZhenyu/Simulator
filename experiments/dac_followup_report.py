"""Render an evidence-bound local follow-up report, without changing measurements."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
from urllib.parse import quote


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(value)


def esc(value):
    return html.escape(str(value), quote=True)


def render(evidence_path, out):
    evidence_path, out = Path(evidence_path).resolve(), Path(out).resolve()
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if evidence.get("schema") != "dac-followup-report-1":
        raise ValueError("Explicit follow-up evidence schema required")
    out.mkdir(parents=True, exist_ok=True)
    bound = []
    for source in evidence["sources"]:
        path = Path(source["path"])
        if not path.is_file() or sha(path) != source["sha256"]:
            raise ValueError("Changed or unavailable evidence: " + str(path))
        bound.append(source)
    blocks = []
    for i, section in enumerate(evidence["sections"], 1):
        blocks.append(f'<section id="study-{i}"><h2>{i}. {esc(section["title"])}</h2>')
        blocks.append(f'<p><strong>本轮结论：</strong>{esc(section["conclusion"])}</p>')
        for paragraph in section.get("paragraphs", []):
            blocks.append("<p>" + esc(paragraph) + "</p>")
        for table in section.get("tables", []):
            headers = table["headers"]
            if any(len(row) != len(headers) for row in table["rows"]):
                raise ValueError("Unequal table row width")
            blocks.append('<div class="table-wrap"><table><thead><tr>')
            blocks.extend("<th>" + esc(value) + "</th>" for value in headers)
            blocks.append("</tr></thead><tbody>")
            for row in table["rows"]:
                blocks.append("<tr>" + "".join("<td>" + esc(value) + "</td>" for value in row) + "</tr>")
            blocks.append("</tbody></table></div>")
        if section.get("limits"):
            blocks.append("<p><strong>仍未证明：</strong>" + esc(section["limits"]) + "</p>")
        for figure in section.get("figures", []):
            source = Path(figure["path"])
            if "fixture" in str(source).lower() or sha(source) != figure["sha256"]:
                raise ValueError("Unverified scientific figure")
            name = f"study-{i}-" + source.name
            target = out / "figures" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            blocks.append('<figure><img src="figures/' + quote(name) + '" alt="' + esc(figure["caption"]) + '">'
                          '<figcaption>' + esc(figure["caption"]) + '</figcaption></figure>')
        blocks.append("</section>")
    links = []
    for item in bound:
        path = Path(item["path"])
        relative = os.path.relpath(path, out).replace("\\", "/")
        label = item.get("label", path.name)
        links.append('<li><a href="' + quote(relative, safe="/:") + '">' + esc(label) + '</a>'
                     '<br><code>' + esc(item["sha256"]) + '</code></li>')
    timestamp = datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")
    content = '''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DAC补充实验报告</title><style>
body{font-family:"Segoe UI","Microsoft YaHei",sans-serif;color:#182027;margin:0;background:#fff;line-height:1.75}
main{max-width:1120px;margin:32px auto;padding:0 24px 60px}h1{font-size:28px;line-height:1.4}h2{font-size:22px;margin-top:36px}
p{margin:12px 0}a{color:#174ea6}small,figcaption{color:#485563}.table-wrap{overflow-x:auto;margin:20px 0}
table{border-collapse:collapse;width:100%;font-size:15px}th,td{border:1px solid #d9dfe4;padding:10px 12px;text-align:left;vertical-align:top}
th{background:#edf2f6}tr:nth-child(even){background:#fafbfc}figure{margin:24px 0}img{max-width:100%;height:auto}
code{font-size:11px;overflow-wrap:anywhere}li{margin:12px 0}.meta{font-size:13px;color:#52616d}
@media print{main{max-width:none;margin:0;padding:0}section{break-inside:auto}a{color:inherit}}
</style><main><h1>DAC补充实验报告</h1>'''
    content += '<p class="meta">生成时间：' + esc(timestamp) + '（北京时间）</p>'
    content += '<p>' + esc(evidence["overview"]) + '</p>'
    content += '<p><strong>历史结果处理：</strong>' + esc(evidence["historical_policy"]) + '</p>'
    content += "".join(blocks)
    content += '<h2>复现与证据</h2><p>' + esc(evidence["reproduction"]) + '</p><ul>' + "".join(links) + '</ul>'
    content += '</main></html>\n'
    write(out / "REPORT_zh.html", content)
    write(out / "evidence.json", json.dumps(evidence, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    receipt = dict(schema="dac-followup-report-receipt-1", generated_at=timestamp,
                   report_sha256=sha(out / "REPORT_zh.html"), source_evidence_sha256=sha(evidence_path),
                   evidence_sources=len(bound), sections=len(evidence["sections"]),
                   report=str(out / "REPORT_zh.html"), authoring_script_sha256=sha(__file__))
    write(out / "report-receipt.json", json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(render(args.evidence, args.out), ensure_ascii=False))
