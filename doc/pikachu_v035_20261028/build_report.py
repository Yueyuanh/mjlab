"""把 report_template.html 里的资源占位符换成 data URI，产出一份自包含的 HTML。

占位符：
  @@IMG:<相对 media/figures 的路径>@@   -> data:image/png;base64,...
  @@VID:<media/web 下的文件名，无扩展>@@ -> data:video/mp4;base64,...

用法：python3 build_report.py [输出路径]
默认输出 reports/index.html（与本目录的模板和 README 放在一起）。
"""

from __future__ import annotations

import base64
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIG = HERE / "media" / "figures"
VID = HERE / "media" / "web"

IMG_RE = re.compile(r"@@IMG:([^@]+)@@")
VID_RE = re.compile(r"@@VID:([^@]+)@@")


def data_uri(path: Path) -> str:
  mime = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".svg": "image/svg+xml",
    ".mp4": "video/mp4",
  }[path.suffix.lower()]
  return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


def main() -> int:
  out = Path(sys.argv[1] if len(sys.argv) > 1 else HERE / "reports" / "index.html")
  html = (HERE / "report_template.html").read_text(encoding="utf-8")

  missing: list[str] = []

  def img(m: re.Match[str]) -> str:
    p = FIG / m.group(1)
    if not p.exists():
      missing.append(f"IMG {m.group(1)}")
      return m.group(0)
    return data_uri(p)

  def vid(m: re.Match[str]) -> str:
    p = VID / f"{m.group(1)}.mp4"
    if not p.exists():
      missing.append(f"VID {m.group(1)}")
      return m.group(0)
    return data_uri(p)

  html = IMG_RE.sub(img, html)
  html = VID_RE.sub(vid, html)

  if missing:
    print("缺失资源：", *missing, sep="\n  ")
    return 1
  if "@@" in html:
    print("仍有未替换的占位符：", re.findall(r"@@[^@]+@@", html))
    return 1

  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(html, encoding="utf-8")
  size = out.stat().st_size / 1e6
  print(f"wrote {out}  ({size:.2f} MB)")
  return 0 if size <= 16 else 1


if __name__ == "__main__":
  raise SystemExit(main())
