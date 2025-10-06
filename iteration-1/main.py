#!/usr/bin/env python3
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import *
import argparse

def main():
    parser = argparse.ArgumentParser(description="Rule-based fraud detection in PySpark")
    parser.add_argument("--csv", required=True, help="Path to transactions CSV")
    parser.add_argument("--flag-threshold", type=int, default=50, help="Risk score threshold to flag")
    parser.add_argument("--high-amount-threshold", type=float, default=2000.0)
    parser.add_argument("--velocity-window-minutes", type=int, default=5)
    parser.add_argument("--velocity-max-allowed", type=int, default=3)
    parser.add_argument("--blacklisted-mcc", type=int, nargs="*", default=[4829, 7995])
    args = parser.parse_args()

    spark = (
        SparkSession.builder
        .appName("RuleBasedFraudDetection")
        .getOrCreate()
    )

    # Schema definition
    schema = StructType([
        StructField("txn_id", StringType()),
        StructField("user_id", StringType()),
        StructField("amount", DoubleType()),
        StructField("currency", StringType()),
        StructField("card_country", StringType()),
        StructField("ip_country", StringType()),
        StructField("merchant_mcc", IntegerType()),
        StructField("device_id", StringType()),
        StructField("timestamp", TimestampType()),
    ])

    df = spark.read.option("header", True).schema(schema).csv(args.csv)

    # ===== RULE 1: High Amount =====
    df = df.withColumn(
        "r_high_amount",
        F.when(F.col("amount") >= args.high_amount_threshold * 2, F.lit(50))
         .when(F.col("amount") >= args.high_amount_threshold, F.lit(30))
         .otherwise(F.lit(0))
    )

    # ===== RULE 2: Country Mismatch =====
    df = df.withColumn(
        "r_country_mismatch",
        F.when((F.col("card_country").isNotNull()) &
               (F.col("ip_country").isNotNull()) &
               (F.col("card_country") != F.col("ip_country")),
               F.lit(40)).otherwise(F.lit(0))
    )

    # ===== RULE 3: Blacklisted MCC =====
    df = df.withColumn(
        "r_blacklisted_mcc",
        F.when(F.col("merchant_mcc").isin(args.blacklisted_mcc), F.lit(25)).otherwise(F.lit(0))
    )

    # ===== RULE 4: High Velocity (window-based) =====
    window_spec = (
        Window.partitionBy("user_id")
        .orderBy(F.col("timestamp").cast("long"))
        .rangeBetween(-args.velocity_window_minutes * 60, 0)
    )

    df = df.withColumn("txn_count_window", F.count("*").over(window_spec))
    df = df.withColumn(
        "r_high_velocity",
        F.when(F.col("txn_count_window") >= args.velocity_max_allowed, F.lit(30)).otherwise(F.lit(0))
    )

    # ===== RULE 5: Night Time =====
    df = df.withColumn(
        "r_night_time",
        F.when((F.hour("timestamp") >= 0) & (F.hour("timestamp") < 5), F.lit(10)).otherwise(F.lit(0))
    )

    # ===== RULE 6: New Device per User =====
    # We'll detect if the device_id is new for the user by using first occurrence
    device_window = Window.partitionBy("user_id", "device_id").orderBy("timestamp")
    df = df.withColumn("first_seen_device", F.min("timestamp").over(device_window))
    df = df.withColumn(
        "r_new_device",
        F.when(F.col("timestamp") > F.col("first_seen_device"), F.lit(0))  # seen before
         .otherwise(F.lit(20))  # first time device appears
    )

    # ===== TOTAL SCORE =====
    rule_cols = ["r_high_amount", "r_country_mismatch", "r_blacklisted_mcc", "r_high_velocity", "r_night_time", "r_new_device"]
    df = df.withColumn("total_score", sum([F.col(c) for c in rule_cols]))

    # ===== SEVERITY LABEL =====
    df = df.withColumn(
        "severity",
        F.when(F.col("total_score") >= 80, F.lit("HIGH"))
         .when(F.col("total_score") >= 50, F.lit("MEDIUM"))
         .when(F.col("total_score") >= 20, F.lit("LOW"))
         .otherwise(F.lit("INFO"))
    )

    # ===== FINAL STATUS =====
    df = df.withColumn(
        "status",
        F.when(F.col("total_score") >= args.flag_threshold, F.lit("FLAG")).otherwise(F.lit("OK"))
    )

    # ===== OUTPUT =====
    df.select(
        "txn_id", "user_id", "amount", "currency", "timestamp",
        "total_score", "severity", "status",
        *rule_cols
    ).show(truncate=False)

    # Optional: write results to CSV/Parquet
    output_path = "output/fraud_scores"
    df.write.mode("overwrite").option("header", True).csv(output_path)

    print(f"\n✅ Results written to {output_path}")

    spark.stop()

if __name__ == "__main__":
    main()
