"""兼容层：旧的 ``tensor_mapping.engine`` 导入路径。

实现已迁到 ``llm_infer_model.tensor.engine``（见
``mapping/ALIGNMENT_IMPLEMENTATION.md`` §3）。这里**只转出**原来的名字，不保留
第二份实现、不重新定义同名数据类——两份定义会让 ``isinstance`` 与
``dataclasses.replace`` 悄悄失效，而这正是兼容层最容易出的错。

``tensor_mapping`` 自己的模块**不**经由本文件导入内核（它们直接从
``llm_infer_model.tensor`` 导入）。本文件只为仓库外的旧调用方、以及按旧路径
书写的测试而存在；一旦没有调用方，它可以整个删掉，不需要动别处。
"""

from __future__ import annotations

from llm_infer_model.tensor.engine import *  # noqa: F401,F403
from llm_infer_model.tensor.engine import __all__  # noqa: F401
