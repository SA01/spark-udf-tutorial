from typing import Dict, List

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructField, StructType, IntegerType, StringType, MapType, LongType, Row, BooleanType
import pyspark.sql.functions as spark_func

from common import create_spark_session


def create_car_sales_data(spark: SparkSession) -> DataFrame:
    data = [
        (1, "Audi", "A8", "2022-11-01", """{"Dealership":3,"Online":5}"""),
        (2, "Audi", "A8", "2022-11-02", """{"Online":4}"""),
        (3, "Audi", "A8", "2022-11-04", """{"Online":1}"""),
        (4, "BMW", "5-series", "2022-11-03", """{"Dealership":2,"Online":10}"""),
        (5, "BMW", "5-series", "2022-11-05", """{"Dealership":3}"""),
        (6, "Toyota", "Camry", "2022-11-02", """{"Dealership":12}"""),
        (7, "Toyota", "Camry", "2022-11-03", """{"Dealership":15}"""),
        (8, "Honda", "Accord", "2022-11-01", """{"Dealership":1,"Online":3}""")
    ]

    data_input_schema = StructType([
        StructField(name="id",dataType=IntegerType()),
        StructField(name="make",dataType=StringType()),
        StructField(name="model",dataType=StringType()),
        StructField(name="date",dataType=StringType()),
        StructField(name="sales",dataType=StringType())
    ])

    sales_schema = MapType(StringType(), LongType())

    data_df = spark\
        .createDataFrame(data=data, schema=data_input_schema)\
        .withColumn("date", spark_func.to_date("date", "yyyy-MM-dd"))\
        .withColumn(
            "sales",
            spark_func.from_json(
                spark_func.col("sales"),
                schema=sales_schema,
                options={"mode":"FAILFAST"}
            )
        )

    return data_df


@spark_func.udf(returnType=StructType(
    fields=[
        StructField(name="dealership", dataType=IntegerType()),
        StructField(name="online", dataType=IntegerType()),
        StructField(name="did_sell_online", dataType=BooleanType()),
    ])
)
def create_struct_from_map(sales_map: Dict[str, int]) -> Row:
    dealership_sales = 0
    online_sales = 0
    did_sell_online = False

    if 'Dealership' in sales_map:
        dealership_sales = sales_map['Dealership']

    if 'Online' in sales_map:
        online_sales = sales_map['Online']
        if online_sales > 0:
            did_sell_online = True

    return Row(dealership=dealership_sales, online=online_sales, did_sell_online=did_sell_online)


@spark_func.udf(returnType=StructType(
    fields=[
        StructField(name="dealership", dataType=IntegerType()),
        StructField(name="online", dataType=IntegerType()),
        StructField(name="did_sell_online", dataType=BooleanType()),
    ])
)
def aggregate_sales(sales: List[Row]) -> Row:
    dealership_sales = sum([item['dealership'] for item in sales])
    online_sales = sum([item['online'] for item in sales])
    did_sell_online = online_sales > 0

    return Row(dealership=dealership_sales, online=online_sales, did_sell_online=did_sell_online)


if __name__ == '__main__':
    spark = create_spark_session(app_name="Maps in UDF")

    test_data = create_car_sales_data(spark=spark)\
        .withColumn("sales", create_struct_from_map("sales"))

    print("After first UDF")
    test_data.show(truncate=False)
    test_data.printSchema()

    result_data = test_data\
        .groupby("make", "model")\
        .agg(spark_func.collect_list("sales").alias("sales"))\
        .withColumn("sales", aggregate_sales("sales"))

    print("Final result")
    result_data.show(truncate=False)
