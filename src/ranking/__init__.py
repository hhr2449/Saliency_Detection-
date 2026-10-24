"""流形排序的对外接口。"""

from .ranking import compute_ranking_matrix, rank_queries

__all__ = ["compute_ranking_matrix", "rank_queries"]
