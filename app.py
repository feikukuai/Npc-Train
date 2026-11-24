"""阿禾：创空间入口。纯上下文整句模型，和 context_app 共用同一个页面。"""
from __future__ import annotations

import os

from expertnet.zh.context_app import build, warmup

CSS = """
.gradio-container { max-width: 900px !important; }
footer { display: none !important; }
"""


demo = build()
warmup()
demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)),
            share=False, css=CSS)
