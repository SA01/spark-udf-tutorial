import time
from datetime import datetime, date
from typing import List, Union

from pyspark.sql import DataFrame, SparkSession
import pyspark.sql.functions as spark_func
from pyspark.sql.types import IntegerType, StructType, StructField, ArrayType, StringType, DateType

from common import create_spark_session
from random import Random


def generate_dates_array_data(spark: SparkSession) -> DataFrame:
    data = [
        (1, ["2022-01-02", "2022-01-08", "2022-01-04", "2022-01-07"]),
        (2, ["2022-01-03", "2022-01-01", "2022-01-02"]),
        (3, ["2022-01-10", "2022-01-12", "2022-01-03", "2022-01-15", "2022-01-01"]),
        (4, ["2022-01-22", "2022-01-21", "2022-01-10", "2022-01-14", "2022-01-24", "2022-01-15", "2022-01-06"]),
        (5, ["2022-01-03"]),
    ]

    data_schema = StructType([
        StructField("id", IntegerType()),
        StructField("dates", ArrayType(StringType())),
    ])

    data_df = spark.createDataFrame(data=data, schema=data_schema)

    return data_df


def generate_minimum_from_exmaple_data(spark: SparkSession) -> DataFrame:
    data = [
        (1, "2022-01-03", ["2022-01-02", "2022-01-08", "2022-01-04", "2022-01-07"]),
        (2, "2022-01-05", ["2022-01-03", "2022-01-01", "2022-01-02"]),
        (3, "2022-01-01", ["2022-01-10", "2022-01-12", "2022-01-03", "2022-01-15", "2022-01-01"]),
        (4, "2022-01-11", ["2022-01-22", "2022-01-21", "2022-01-10", "2022-01-14", "2022-01-24", "2022-01-15", "2022-01-06"]),
        (5, "2022-01-03", ["2022-01-03"]),
    ]

    data_schema = StructType([
        StructField("id", IntegerType()),
        StructField("minimum_after", StringType()),
        StructField("dates", ArrayType(StringType())),
    ])

    data_df = spark\
        .createDataFrame(data=data, schema=data_schema)\
        .withColumn("minimum_after", spark_func.to_date("minimum_after", "yyyy-MM-dd"))

    return data_df


@spark_func.udf(returnType=DateType())
def minimum_date(date_strs: List[str]) -> date:
    return min([datetime.strptime(dt_str, "%Y-%m-%d") for dt_str in date_strs]).date()


@spark_func.udf(returnType=DateType())
def minimum_date_after(date_strs: List[str], minimum_after: date) -> Union[date, None]:
    filtered_dates = [
        datetime.strptime(dt_str, "%Y-%m-%d").date() for dt_str in date_strs
        if datetime.strptime(dt_str, "%Y-%m-%d").date() >= minimum_after
    ]

    if filtered_dates:
        return min(filtered_dates)
    else:
        return None


if __name__ == '__main__':
    spark = create_spark_session(app_name="Arrays in UDF")

    # Example 1 - minimum date in array of dates
    test_df = generate_dates_array_data(spark=spark)
    result_df = test_df.withColumn("minimum_date", minimum_date(spark_func.col("dates")))
    result_df.show(truncate=False)
    result_df.printSchema()

    # Example 1 using spark functions
    result_df_alt = test_df.withColumn(
        "minimum_date",
        spark_func.array_min(spark_func.transform("dates", lambda dt_str: spark_func.to_date(dt_str, "yyyy-MM-dd"))))
    result_df_alt.show(truncate=False)

    # Example 2 - minimum date after given date in array of dates
    test_df = generate_minimum_from_exmaple_data(spark=spark)
    result_df = test_df.withColumn(
        "minimum_date", minimum_date_after(spark_func.col("dates"), spark_func.col("minimum_after"))
    )
    result_df.show(truncate=False)

    # Performance UDF vs. spark sql functions

    start = time.time()
    test_df = spark.read.parquet("test_data/*.parquet")
    print(f"Data contains {test_df.count()} rows")

    # Comment out one and run other, compare over multiple runs
    # Using Spark UDF
    min_dates_df = test_df\
        .withColumn("minimum_date", minimum_date(spark_func.col("dates")))\
        .groupby("minimum_date")\
        .agg(spark_func.count(spark_func.lit(1)).alias("num_rows"))\
        .show(truncate=False, n=1000)

    # Using Spark SQL
    min_dates_df = test_df.withColumn(
        "minimum_date",
        spark_func.array_min(spark_func.transform("dates", lambda dt_str: spark_func.to_date(dt_str, "yyyy-MM-dd")))
    )\
        .groupby("minimum_date")\
        .agg(spark_func.count(spark_func.lit(1)).alias("num_rows"))\
        .show(truncate=False, n=1000)

    print(f"Took {time.time() - start} seconds.")

