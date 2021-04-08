# -*- coding: utf-8 -*-

# Import libraries
from pyathena import connect
from pyathena.pandas.util import as_pandas
import datetime
import numpy as np
import pandas as pd
import os
import boto3
from botocore.client import ClientError

s3 = boto3.resource('s3')
client = boto3.client("sts")
account_id = client.get_caller_identity()["Account"]
my_session = boto3.session.Session()
region = my_session.region_name
athena_query_results_bucket = 'aws-athena-query-results-' + account_id + '-' + region

try:
    s3.meta.client.head_bucket(Bucket=athena_query_results_bucket)
except ClientError:
    bucket = s3.create_bucket(Bucket=athena_query_results_bucket)
    print('Creating bucket ' + athena_query_results_bucket)
cursor = connect(s3_staging_dir='s3://' + athena_query_results_bucket + '/athena/temp').cursor()


## Cohort Construction:

# Select microbiology tests associated with ESBL
# producing bacteria from microbiologyevents table and
# link them with patients admission date from
# mimiciii.admissions table:

def query_esbl_pts():
    """
    Query to select esbl microbiology tests
    and link them to patient's admission time
    """
    query = """
WITH admissions AS (
SELECT
    admits.subject_id,
    admits.hadm_id,
    admits.admittime,
    admits.deathtime, 
    microb.charttime,
    date_diff('hour', admits.admittime, microb.charttime) as diff,
    CASE WHEN admits.deathtime < microb.charttime THEN 1 ELSE 0 END AS death_before_rslt,
    CASE WHEN date_diff('hour', admits.admittime, microb.charttime) IS NULL THEN '<=00h'
         WHEN date_diff('hour', admits.admittime, microb.charttime) <= 0  THEN '<=00h'
         WHEN date_diff('hour', admits.admittime, microb.charttime) <= 6  THEN '<=06h'
         WHEN date_diff('hour', admits.admittime, microb.charttime) <= 12 THEN '<=12h'
         WHEN date_diff('hour', admits.admittime, microb.charttime) <= 24 THEN '<=24h'
         WHEN date_diff('hour', admits.admittime, microb.charttime) <= 36 THEN '<=36h'
         WHEN date_diff('hour', admits.admittime, microb.charttime) <= 48 THEN '<=48h'
    ELSE '>48h' END AS time_to_rslt,
    microb.org_itemid, 
    microb.org_name,
    CASE WHEN microb.interpretation in ('R','I') THEN 1 ELSE 0 END AS RESISTANT_YN,
    CASE WHEN microb.interpretation = 'S' THEN 1 ELSE 0 END SENSITIVE_YN
FROM mimiciii.admissions admits
INNER JOIN mimiciii.microbiologyevents microb
    ON microb.hadm_id = admits.hadm_id 
WHERE ab_name in ('CEFTAZIDIME') AND
      org_itemid in (
                80004, -- KLEBSIELLA PNEUMONIAE
                80026, -- PSEUDOMONAS AERUGINOSA
                80005, -- KLEBSIELLA OXYTOCA
                80017, -- PROTEUS MIRABILIS
                80040, -- NEISSERIA GONORRHOEAE
                80008, -- ENTEROBACTER CLOACAE
                80007, -- ENTEROBACTER AEROGENES
                80002) -- ESCHERICHIA COLI
    )
SELECT         
    admissions.subject_id,
    admissions.hadm_id,
    admissions.admittime,
    admissions.charttime,
    admissions.diff,
    admissions.time_to_rslt,
    RESISTANT_YN,
    SENSITIVE_YN
FROM admissions
WHERE admissions.time_to_rslt <> '<=00h' AND 
      admissions.death_before_rslt != 1 

"""
    cursor.execute(query)
    df = as_pandas(cursor)

    return df


def remove_dups(df):
    """
    When more than one microbiology test exists keep first.
    In case when there are multiple tests use the same timestamp then
    keep the one with 'RESISTANT_YN' = 1 or first available record
    when all 'RESISTANT_YN' = 0.
    """
    # Sort values
    df = df.sort_values(by=['subject_id','hadm_id','admittime','charttime','RESISTANT_YN'],
                       ascending = [True, True, True, True, False])
    # Remove duplicates
    df = df.drop_duplicates(subset=['subject_id','hadm_id','admittime'], keep='first')
    return df

def observation_window(df, window_size):
    """
    Create an index date by adding time window in hours
    to admission time. Select records when the index date
    is before microbiology test results.
    """
    # Index date equals to admit time plus selected observation window
    df['index_date'] = df['admittime'] + pd.to_timedelta(window_size, unit='h')
    # Exclude cases when the diagnosis of resistant bacteria is earlier than index_date
    subset = df[df['index_date'] < df['charttime']]
    return subset