import polars as pl


def estimate_yaw(df: pl.DataFrame) -> pl.DataFrame:
    # TODO estimate correctly
    return df.with_columns(pl.lit(0.0).alias("yaw"))
