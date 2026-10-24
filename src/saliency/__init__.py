"""背景查询与前景查询的对外接口。"""

from .background import background_query
from .foreground import foreground_query

__all__ = ["background_query", "foreground_query"]
