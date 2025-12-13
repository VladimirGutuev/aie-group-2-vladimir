from __future__ import annotations

import pandas as pd

from eda_cli.core import (
    compute_quality_flags,
    correlation_matrix,
    flatten_summary_for_print,
    missing_table,
    summarize_dataset,
    top_categories,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [10, 20, 30, None],
            "height": [140, 150, 160, 170],
            "city": ["A", "B", "A", None],
        }
    )


def test_summarize_dataset_basic():
    df = _sample_df()
    summary = summarize_dataset(df)

    assert summary.n_rows == 4
    assert summary.n_cols == 3
    assert any(c.name == "age" for c in summary.columns)
    assert any(c.name == "city" for c in summary.columns)

    summary_df = flatten_summary_for_print(summary)
    assert "name" in summary_df.columns
    assert "missing_share" in summary_df.columns


def test_missing_table_and_quality_flags():
    df = _sample_df()
    missing_df = missing_table(df)

    assert "missing_count" in missing_df.columns
    assert missing_df.loc["age", "missing_count"] == 1

    summary = summarize_dataset(df)
    flags = compute_quality_flags(summary, missing_df)
    assert 0.0 <= flags["quality_score"] <= 1.0


def test_correlation_and_top_categories():
    df = _sample_df()
    corr = correlation_matrix(df)
    # корреляция между age и height существует
    assert "age" in corr.columns or corr.empty is False

    top_cats = top_categories(df, max_columns=5, top_k=2)
    assert "city" in top_cats
    city_table = top_cats["city"]
    assert "value" in city_table.columns
    assert len(city_table) <= 2


# ========================
# Тесты для новых эвристик
# ========================


def test_has_constant_columns():
    """Тест эвристики has_constant_columns: проверяем обнаружение константных колонок."""
    # DataFrame с константной колонкой
    df_with_const = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "constant_col": ["same", "same", "same", "same"],
            "varying_col": ["a", "b", "c", "d"],
        }
    )

    summary = summarize_dataset(df_with_const)
    missing_df = missing_table(df_with_const)
    flags = compute_quality_flags(summary, missing_df, df=df_with_const)

    # Должен обнаружить константную колонку
    assert flags["has_constant_columns"] is True
    assert "constant_col" in flags["constant_columns"]
    assert "varying_col" not in flags["constant_columns"]


def test_has_constant_columns_none():
    """Тест: датасет без константных колонок."""
    df_no_const = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": ["x", "y", "z"],
        }
    )

    summary = summarize_dataset(df_no_const)
    missing_df = missing_table(df_no_const)
    flags = compute_quality_flags(summary, missing_df, df=df_no_const)

    assert flags["has_constant_columns"] is False
    assert flags["constant_columns"] == []


def test_has_high_cardinality_categoricals():
    """Тест эвристики has_high_cardinality_categoricals."""
    # Создаём DataFrame с категориальной колонкой высокой кардинальности
    high_card_values = [f"val_{i}" for i in range(60)]
    df_high_card = pd.DataFrame(
        {
            "id": list(range(60)),
            "high_card_cat": high_card_values,
            "low_card_cat": ["a", "b"] * 30,
        }
    )

    summary = summarize_dataset(df_high_card)
    missing_df = missing_table(df_high_card)
    # Порог по умолчанию 50
    flags = compute_quality_flags(summary, missing_df, df=df_high_card, high_cardinality_threshold=50)

    assert flags["has_high_cardinality_categoricals"] is True
    assert "high_card_cat" in flags["high_cardinality_columns"]
    assert "low_card_cat" not in flags["high_cardinality_columns"]


def test_has_many_zero_values():
    """Тест эвристики has_many_zero_values: проверяем обнаружение колонок с большой долей нулей."""
    df_with_zeros = pd.DataFrame(
        {
            "mostly_zeros": [0, 0, 0, 0, 0, 0, 1, 2],  # 75% нулей
            "few_zeros": [1, 2, 3, 4, 5, 0, 7, 8],     # 12.5% нулей
            "no_zeros": [1, 2, 3, 4, 5, 6, 7, 8],      # 0% нулей
        }
    )

    summary = summarize_dataset(df_with_zeros)
    missing_df = missing_table(df_with_zeros)
    flags = compute_quality_flags(summary, missing_df, df=df_with_zeros, zero_share_threshold=0.5)

    assert flags["has_many_zero_values"] is True
    assert "mostly_zeros" in flags["zero_heavy_columns"]
    assert "few_zeros" not in flags["zero_heavy_columns"]
    assert "no_zeros" not in flags["zero_heavy_columns"]
    assert flags["max_zero_share"] == 0.75  # 6 из 8 = 75%


def test_quality_score_with_new_flags():
    """Тест: quality_score должен снижаться при наличии проблем."""
    # Датасет без проблем
    df_good = pd.DataFrame(
        {
            "a": list(range(100)),
            "b": list(range(100, 200)),
        }
    )

    # Датасет с константной колонкой
    df_with_const = pd.DataFrame(
        {
            "a": list(range(100)),
            "constant": ["same"] * 100,
        }
    )

    summary_good = summarize_dataset(df_good)
    missing_good = missing_table(df_good)
    flags_good = compute_quality_flags(summary_good, missing_good, df=df_good)

    summary_bad = summarize_dataset(df_with_const)
    missing_bad = missing_table(df_with_const)
    flags_bad = compute_quality_flags(summary_bad, missing_bad, df=df_with_const)

    # Скор "хорошего" датасета должен быть выше
    assert flags_good["quality_score"] >= flags_bad["quality_score"]
