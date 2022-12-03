from typing import List, Dict

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import StructField, StringType, StructType, IntegerType, MapType, LongType, Row
import pyspark.sql.functions as spark_func

from common import create_spark_session


def create_car_sales_data(spark: SparkSession) -> DataFrame:
    data = [
        (1, "Audi", "A8", "2022-11-01", """{"Boston":5,"New York":3}"""),
        (2, "Audi", "A8", "2022-11-02", """{"New York":1}"""),
        (3, "Audi", "A8", "2022-11-04", """{"Philadelphia":2,"San Francisco":3}"""),
        (4, "BMW", "5-series", "2022-11-02", """{"Boston":1,"San Francisco":2,"Philadelphia":2}"""),
        (5, "BMW", "5-series", "2022-11-03", """{"Minneapolis":3}"""),
        (6, "BMW", "5-series", "2022-11-05", """{"Boston":5,"Minneapolis":5,"Philadelphia":1}"""),
        (7, "BMW", "5-series", "2022-11-06", """{"Philadelphia":3}"""),
        (8, "Toyota", "Camry", "2022-11-02", """{"Boston":8,"New York":6,"Philadelphia":4,"San Francisco":7}"""),
        (9, "Toyota", "Camry", "2022-11-03", """{"Boston":9,"New York":3,"San Francisco":6}"""),
        (10, "Honda", "Accord", "2022-11-01", """{"Boston":8,"New York":7}""")
    ]

    data_input_schema = StructType([
        StructField(name="id",dataType=IntegerType()),
        StructField(name="make",dataType=StringType()),
        StructField(name="model",dataType=StringType()),
        StructField(name="date",dataType=StringType()),
        StructField(name="sales_by_city",dataType=StringType())
    ])

    sales_by_city_schema = MapType(StringType(), LongType())

    data_df = spark\
        .createDataFrame(data=data, schema=data_input_schema)\
        .withColumn("date", spark_func.to_date("date", "yyyy-MM-dd"))\
        .withColumn(
            "sales_by_city",
            spark_func.from_json(
                spark_func.col("sales_by_city"),
                schema=sales_by_city_schema,
                options={"mode":"FAILFAST"}
            )
        )

    return data_df


@spark_func.udf(returnType=MapType(StringType(), LongType()))
def aggregate_maps(input_maps: List[Dict[str, int]]) -> Dict[str, int]:
    agg_result = {}
    for map in input_maps:
        for (key, value) in map.items():
            if key in agg_result:
                agg_result[key] = agg_result[key] + value
            else:
                agg_result[key] = value

    return agg_result


if __name__ == '__main__':
    spark = create_spark_session(app_name="Maps in UDF")

    data_df = create_car_sales_data(spark=spark)
    data_df.show(truncate=False)
    aggregated_df = data_df\
        .groupby("make", "model")\
        .agg(spark_func.collect_list("sales_by_city").alias("sales_by_city"))

    result_df = aggregated_df.withColumn("agg_sales", aggregate_maps("sales_by_city"))

    result_df.show(truncate=False)
