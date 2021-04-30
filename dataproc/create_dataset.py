# Import libraries
from pyathena import connect
from pyathena.pandas.cursor import PandasCursor
from pyathena.pandas.util import as_pandas
import boto3
from botocore.client import ClientError

s3 = boto3.resource('s3')
client = boto3.client("sts")
account_id = client.get_caller_identity()["Account"]
my_session = boto3.session.Session()
region = my_session.region_name
athena_query_results_bucket = 'aws-athena-query-results-'+account_id+'-'+region

try:
    s3.meta.client.head_bucket(Bucket=athena_query_results_bucket)
except ClientError:
    bucket = s3.create_bucket(Bucket=athena_query_results_bucket)
    print('Creating bucket '+athena_query_results_bucket)
cursor = connect(s3_staging_dir='s3://'+athena_query_results_bucket+'/athena/temp').cursor(PandasCursor)

# The above code comes directly from aline-awsathena.ipynb in the MIMIC-III starter code

def lab_events(hadm_ids_table: str):
#     hadm_ids = ','.join(map(str, hadm_ids))
    statement = f"""
    SELECT   E.hadm_id,
             E.itemid,
             case 
                 -- use valuenum if provided, otherwise the textual. See https://mimic.physionet.org/mimictables/labevents/
                 when E.valuenum is not null then cast(E.valuenum as varchar)
                 else E.value
             end value,
             E.flag,
             E.charttime
             -- I.label,
             -- I.fluid,
             -- I.category,
             -- I.loinc_code
    FROM mimiciii.labevents E
    LEFT JOIN mimiciii.d_labitems I
        ON E.itemid=I.itemid
    JOIN {hadm_ids_table} admissions_list 
        on admissions_list.subject_id = E.subject_id
        AND E.charttime <=  admissions_list.index_date
        AND date_diff('day',E.charttime , admissions_list.admittime) <= 7
    WHERE  (lower(I.fluid) LIKE '%blood%'
            OR lower(I.fluid) LIKE '%urine%')
    """
    df = cursor.execute(statement).as_pandas()
    return df

def important_labs():
    # find the lab codes which can be found in first hours, taken from blood or urine
    statement = """
    with admissions AS
        (SELECT hadm_id,
             admittime
        FROM
            (SELECT admissions.hadm_id,
             admissions.admittime,
             cast(min(floor( date_diff('second', admissions.admittime, tests.charttime) / 3600)) AS INTEGER) hours

            FROM mimiciii.microbiologyevents tests
            JOIN mimiciii.admissions admissions
                ON admissions.hadm_id = tests.hadm_id
            WHERE 1=1
                    AND tests.ab_name IN ('CEFTAZIDIME')
                    AND org_itemid IN ( 80004, -- KLEBSIELLA PNEUMONIAE
             80026, -- PSEUDOMONAS AERUGINOSA
             80005, -- KLEBSIELLA OXYTOCA
             80017, -- PROTEUS MIRABILIS
             80040, -- NEISSERIA GONORRHOEAE
             80008,-- ENTEROBACTER CLOACAE
             80007,-- ENTEROBACTER AEROGENES
             80002 --ESCHERICHIA COLI
             )
                    AND charttime is NOT null
            GROUP BY  1,2 ) a
            WHERE a.hours < 12)
        SELECT lab_items.itemid,
             lab_items.category,
             lab_items.label,
             lab_items.fluid,
             lab_items.loinc_code,
             count(*) cnt
    FROM mimiciii.labevents lab
    JOIN mimiciii.d_labitems lab_items
        ON lab.itemid=lab_items.itemid
    JOIN admissions
        ON lab.hadm_id=admissions.hadm_id
    WHERE lab.charttime <= admissions.admittime + interval '24' hour
            AND (lower(lab_items.fluid) LIKE '%blood%'
            OR lower(lab_items.fluid) LIKE '%urine%')
    GROUP BY  1,2,3,4,5
    ORDER BY  cnt DESC;
    """
    df = cursor.execute(statement).as_pandas()
    return df

def static_data(hadm_ids_table: str):
    #hadm_ids = ','.join(map(str, hadm_ids))
    statement = f"""
    SELECT   A.hadm_id,
             A.admittime,
             A.admission_type,
             A.admission_location,
             A.insurance,
             A.language,
             A.religion,
             A.marital_status,
             A.ethnicity,
             P.gender,
             date_diff('year', P.dob, A.admittime) age
    FROM mimiciii.admissions A
    JOIN {hadm_ids_table} addmissions_list on addmissions_list.hadm_id = A.hadm_id
    LEFT JOIN mimiciii.patients P
        ON A.subject_id=P.subject_id
    """
    df = cursor.execute(statement).as_pandas()
    return df

def dataset_creation(hadm_ids: list, observation_window_hours: float):
    static_df = static_data(hadm_ids)
    lab_df = lab_events(hadm_ids, observation_window_hours)
    ilabs_df = important_labs()
    df = lab_df.merge(ilabs_df, how='right', on=['itemid','label','fluid','category','loinc_code'])
    df = df.drop(df.index[df['cnt'] < 100].tolist())
    df = df.drop(columns=['itemid','valueuom','flag','label','fluid','category','cnt'])
    df = df.sort_values('charttime', axis=0)
    df = df.reset_index(drop=True)
    df = df.drop(columns='charttime')
    df = df.drop_duplicates(subset=['subject_id','hadm_id','loinc_code'], keep='first', ignore_index=True)
    df = df.dropna(subset=['loinc_code'])
    df = df.reset_index(drop=True)
    df = df.pivot(['hadm_id', 'subject_id'],'loinc_code','value')
    df = df.reset_index().dropna(subset=['hadm_id'])
    df.columns.name = None
    df = df.merge(static_df,how='left', on=['hadm_id','subject_id'])
    return df
