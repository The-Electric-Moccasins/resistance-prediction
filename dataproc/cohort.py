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

def query_esbl_pts(observation_window_hours):
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
    CASE WHEN admits.deathtime < microb.charttime THEN 1 ELSE 0 END AS death_before_rslt,
    date_diff('hour', admits.admittime, microb.charttime) AS time_to_rslt,
    CASE WHEN microb.interpretation in ('R','I') THEN 1 ELSE 0 END AS RESISTANT_YN,
    CASE WHEN microb.interpretation = 'S' THEN 1 ELSE 0 END SENSITIVE_YN
FROM mimiciii.admissions admits
INNER JOIN mimiciii.microbiologyevents microb
    ON microb.hadm_id = admits.hadm_id 
WHERE ab_name in 
    -- ('CEFTAZIDIME') 
    ('CEFAZOLIN', 'CEFEPIME', 'CEFPODOXIME',
                  'CEFTAZIDIME', 'CEFTRIAXONE', 'CEFUROXIME')
    AND
      org_itemid in (
                80004, -- KLEBSIELLA PNEUMONIAE
                80026, -- PSEUDOMONAS AERUGINOSA
                80005, -- KLEBSIELLA OXYTOCA
                80017, -- PROTEUS MIRABILIS
                80040, -- NEISSERIA GONORRHOEAE
                80008, -- ENTEROBACTER CLOACAE
                80007, -- ENTEROBACTER AEROGENES
                80002 -- ESCHERICHIA COLI
                )
    )
SELECT         
    admissions.subject_id,
    admissions.hadm_id,
    admissions.admittime,
    admissions.charttime,
    admissions.time_to_rslt,
    RESISTANT_YN,
    SENSITIVE_YN
FROM admissions
WHERE admissions.time_to_rslt is not null
      and admissions.time_to_rslt > %(time_window_hours)d

"""
    params = {
        'time_window_hours': observation_window_hours
    }
    cursor.execute(query, params)
    df = as_pandas(cursor)

    return df


def query_esbl_bacteria_label(observation_window_hours):
    query = """
    select hadm_id, max(RESISTANT_BACT) resistant_label from (
WITH admissions AS (
SELECT
    admits.subject_id,
    admits.hadm_id,
    admits.admittime,
    admits.deathtime, 
    microb.charttime,
    CASE WHEN admits.deathtime < microb.charttime THEN 1 ELSE 0 END AS death_before_rslt,
    date_diff('hour', admits.admittime, microb.charttime) AS time_to_rslt,
    CASE WHEN microb.interpretation in ('R','I') THEN 1 ELSE 0 END AS RESISTANT_YN,
    CASE WHEN microb.interpretation in ('R','I','S') THEN org_itemid ELSE 0 END AS RESISTANT_BACT,
    CASE WHEN microb.interpretation in ('S') THEN org_itemid ELSE 0 END AS SENSITIVE_BACT

FROM mimiciii.admissions admits
INNER JOIN mimiciii.microbiologyevents microb
    ON microb.hadm_id = admits.hadm_id 
WHERE ab_name in 
    -- ('CEFTAZIDIME') 
    ('CEFAZOLIN', 'CEFEPIME', 'CEFPODOXIME', 'CEFTAZIDIME', 'CEFTRIAXONE', 'CEFUROXIME')
    AND
      org_itemid in (
                80004, -- KLEBSIELLA PNEUMONIAE
                80026, -- PSEUDOMONAS AERUGINOSA
                80005, -- KLEBSIELLA OXYTOCA
                80017, -- PROTEUS MIRABILIS
                80040, -- NEISSERIA GONORRHOEAE
                80008, -- ENTEROBACTER CLOACAE
                80007, -- ENTEROBACTER AEROGENES
                80002 -- ESCHERICHIA COLI
                )
    )
SELECT         
    admissions.subject_id,
    admissions.hadm_id,
    admissions.admittime,
    admissions.charttime,
    admissions.time_to_rslt,
    RESISTANT_YN,
    RESISTANT_BACT,
    SENSITIVE_BACT
FROM admissions
WHERE admissions.time_to_rslt is not null
      and admissions.time_to_rslt > %(time_window_hours)d

) a
group by 1
    """
    params = {
        'time_window_hours': observation_window_hours
    }
    cursor.execute(query, params)
    df = as_pandas(cursor)

    return df



def query_pts_multi_bacteria(observation_window_hours):
    """
    Query to select multi bacteria microbiology tests
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
    microb.org_itemid org_id,
    CASE WHEN admits.deathtime < microb.charttime THEN 1 ELSE 0 END AS death_before_rslt,
    date_diff('hour', admits.admittime, microb.charttime) AS time_to_rslt
FROM mimiciii.admissions admits
INNER JOIN mimiciii.microbiologyevents microb
    ON microb.hadm_id = admits.hadm_id 
WHERE microb.spec_itemid is not null
        and charttime is not null
        and (
                 org_itemid in (
                80293, -- positive for MRSA
                80004, -- KLEBSIELLA PNEUMONIAE
                80026, -- PSEUDOMONAS AERUGINOSA
                80005, -- KLEBSIELLA OXYTOCA
                80017, -- PROTEUS MIRABILIS
                80040, -- NEISSERIA GONORRHOEAE
                80008, -- ENTEROBACTER CLOACAE
                80007, -- ENTEROBACTER AEROGENES
                80002
                )
                or (spec_type_desc = 'MRSA SCREEN' and org_itemid is null) -- negative for mrsa
             )
  )
SELECT         
    admissions.subject_id,
    admissions.hadm_id,
    admissions.admittime,
    admissions.charttime,
    admissions.time_to_rslt,
    org_id
FROM admissions
WHERE admissions.time_to_rslt is not null
      and admissions.time_to_rslt > %(time_window_hours)d

"""
    params = {
        'time_window_hours': observation_window_hours
    }
    cursor.execute(query, params)
    df = as_pandas(cursor)

    return df


def query_all_pts(observation_window_hours):
    """
    load all patients that are possible for consideration
    
    and link them to patient's admission time
    """
    query = """
    WITH admissions AS (
    SELECT
        admits.subject_id,
        admits.hadm_id,
        admits.admittime,
        admits.deathtime, 
        (admits.admittime + interval %(time_window_hours)s hour + interval '1' hour) charttime,
        (admits.admittime + interval %(time_window_hours)s hour) as diff,
        CASE WHEN admits.deathtime < (admits.admittime + interval %(time_window_hours)s hour) THEN 1 ELSE 0 END AS death_before_rslt,
        1 time_to_rslt,
        null org_itemid, 
        null org_name,
        0 RESISTANT_YN,
        1 SENSITIVE_YN
    FROM mimiciii.admissions admits
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
    WHERE 
          admissions.death_before_rslt != 1 
    order by random()
    limit 10000
    
    """
    params = {
        'time_window_hours': str(observation_window_hours)
    }
    cursor.execute(query, params)
    df = as_pandas(cursor)

    return df



def create_all_pts_within_observation_window(observation_window_hours) -> str:
    """
    create a view of all patients within observation window
    return the view name
    """
    
    view_name = f"default.all_pts_{observation_window_hours}_hours"
    
    query = f"""
    CREATE OR REPLACE VIEW {view_name} AS
    WITH admits AS (
    SELECT
        admits.subject_id,
        admits.hadm_id,
        admits.admittime,
        admits.admittime + interval %(time_window_hours)s hour index_date,
        CASE WHEN admits.deathtime <= (admits.admittime + interval %(time_window_hours)s hour) THEN 1 ELSE 0 END AS death_during_obs_win,
        CASE WHEN admits.dischtime <= (admits.admittime + interval %(time_window_hours)s hour) THEN 1 ELSE 0 END AS disch_during_obs_win
    FROM mimiciii.admissions admits
    )
    SELECT         
        admits.subject_id,
        admits.hadm_id,
        admits.index_date,
        admits.admittime
    FROM admits
    WHERE 
          admits.death_during_obs_win != 1 
          and admits.disch_during_obs_win != 1
    order by random()
    --limit 1000
    
    """
    params = {
        'time_window_hours': str(observation_window_hours)
    }
    cursor.execute(query, params)
    return view_name
 


def query_all_pts_within_observation_window(observation_window_hours):
    """
    Query to select all patients
    and link them to patient's admission time
    """
    table_name = create_all_pts_within_observation_window(observation_window_hours)
    query = f"select * from {table_name}"
    cursor.execute(query)
    df = as_pandas(cursor)
    return df, table_name


def remove_dups(df):
    """
    If test has multiple records with the same 'charttime' then
    choose max value for 'RESISTANT_YN' column
    """
    # # Sort values
    # df = df.sort_values(by=['subject_id','hadm_id','admittime','charttime','RESISTANT_YN'],
    #                    ascending = [True, True, True, True, False])
    # # Remove duplicates
    # df = df.drop_duplicates(subset=['subject_id','hadm_id','admittime'], keep='first')
    
    # Groupby admission id, admit time and chart time to find ristance max
    df = df.groupby(['subject_id','hadm_id','admittime','charttime']).agg({'RESISTANT_YN':'max'}).reset_index()
    
    return df


def remove_dups_multi_label(df):
    """
    remove duplicate label. Keep one label per admission.
    """
    # Sort values
    df = df.sort_values(by=['subject_id','hadm_id','org_id'],
                       ascending = [True, True, True])
    # Remove duplicates
    df = df.drop_duplicates(subset=['subject_id','hadm_id','org_id'], keep='first')
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
    # Keep one label per patient
    subset = subset.groupby(['subject_id','hadm_id','admittime','index_date']).agg({'RESISTANT_YN':'max'}).reset_index()
    
    return subset