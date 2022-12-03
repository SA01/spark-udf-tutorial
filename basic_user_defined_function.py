import time

import pyspark.sql.functions
from pyspark.sql import DataFrame
from pyspark.sql.functions import udf, count, lit
from pyspark.sql.types import IntegerType

from common import create_spark_session
from random import Random


@udf
def random_int_generator() -> int:
    return Random().randint(a=1, b=10)


random_int_generator_alternate = udf(lambda: Random().randint(a=1, b=10), IntegerType()).asNondeterministic()


def create_test_df() -> DataFrame:
    return spark.createDataFrame(
        data=[1, 2, 3, 4, 5],
        schema=IntegerType()
    )


if __name__ == '__main__':
    spark = create_spark_session(app_name="Basic UDF")

    test_df = create_test_df()
    after_udf = test_df.withColumn("random", random_int_generator_alternate()).checkpoint(eager=True)

    after_udf.show(truncate=False)
    after_udf.show(truncate=False)

    grpuped_df = after_udf.groupby("random").agg(count(lit(1)))
    grpuped_df.show(truncate=False)
    grpuped_df.show(truncate=False)
