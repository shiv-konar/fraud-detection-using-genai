from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql import Window as W
import pyspark
import datetime
import json

spark = SparkSession.builder.appName('run-pyspark-code').getOrCreate()


def etl(rp_employees, rp_payroll):
    # Write code here
    e = rp_employees.alias("e")
    p = rp_payroll.alias("p")
    joined_df = e.join(p, e.employee_id == p.employee_id)
    output_df = joined_df.withColumn(
        "pay",
        F.when(F.col("p.hours_worked") <= 40, F.col("p.hours_worked") * F.col("p.hourly_rate"))
        .when(F.col("p.hours_worked") > 40, (F.col("p.hours_worked") - 40) * 1.5 * F.col("p.hourly_rate") + 40 * F.col("p.hourly_rate"))
    ).select("e.employee_id", "e.name", "pay", "p.position")

    return output_df