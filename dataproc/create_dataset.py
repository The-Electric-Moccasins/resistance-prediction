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


def prescriptions(hadm_ids_table: str):
    """
    Query to select antibiotics prescriptions 
    records based patients' hadm_id
    """
#     hadm_ids = ', '.join(map(str, hadm_ids))
#     time_window_hours = str(observation_window_hours)
    print(f"hadm_ids table: {hadm_ids_table}")
    query = f"""
        SELECT 
        admissions.hadm_id,
        antibiotics.startdate,
        antibiotics.enddate,
        antibiotics.drug,
        admissions.admittime,
        admissions_list.index_date
        
    FROM mimiciii.admissions admissions
    JOIN mimiciii.prescriptions antibiotics
        ON antibiotics.hadm_id = admissions.hadm_id
    JOIN {hadm_ids_table} admissions_list 
        on admissions_list.hadm_id = admissions.hadm_id
            AND antibiotics.startdate <= admissions_list.index_date
    WHERE  
        drug like '%Amikacin%' OR drug like '%Ampicillin%' OR
        drug like '%Cefazolin%' OR drug like '%Cefepime%' OR
        drug like '%Cefpodoxime%' OR drug like '%Ceftazidime%' OR
        drug like '%Ceftriaxone%' OR drug like '%Cefuroxime%' OR
        drug like '%Chloramphenicol%' OR drug like '%Ciprofloxacin%' OR
        drug like '%Clindamycin%' OR drug like '%Daptomycin%' OR
        drug like '%Erythromycin%' OR drug like '%Gentamicin%' OR
        drug like '%Imipenem%' OR drug like '%Levofloxacin%' OR
        drug like '%Linezolid%' OR drug like '%Meropenem%' OR
        drug like '%Nitrofurantoin%' OR drug like '%Oxacillin%' OR
        drug like '%Penicillin%' OR drug like '%Piperacillin%' OR
        drug like '%Rifampin%' OR drug like '%Tetracycline%' OR
        drug like '%Tobramycin%' OR drug like '%Trimethoprim%' OR
        drug like '%Sulfonamide%' OR drug like '%Vancomycin%' OR
        drug like '%Tazobactam%'
    
    """
    
    
    df = cursor.execute(query).as_pandas()

    return df


def previous_admissions(hadm_ids_table: str):
    """
    Query to select previous admissions 
    records based patients' hadm_id
    """
#     hadm_ids = ','.join(map(str, hadm_ids)) 
    query = f"""
    SELECT
        admits.hadm_id,
        admits.admittime,
        admits2.hadm_id as prev_hadm_id,
        admits2.admittime as prev_admittime
    FROM mimiciii.admissions admits
    JOIN {hadm_ids_table} addmissions_list 
        on addmissions_list.hadm_id = admits.hadm_id
    LEFT JOIN mimiciii.admissions admits2
    ON admits.subject_id = admits2.subject_id
    WHERE 
        admits2.admittime < admits.admittime AND
        admits2.admittime >= admits.admittime - interval '360' day
        """
    df = cursor.execute(query).as_pandas()
    return df


def open_wounds_diags(hadm_ids_table: str):
    """
    Query to select diagnosis of open wound
    records based patients' hadm_id
    """
#     hadm_ids = ','.join(map(str, hadm_ids)) 
    query = f"""
    SELECT 
        diagnoses_icd.hadm_id,
        diagnoses_icd.icd9_code
    FROM mimiciii.diagnoses_icd
    JOIN {hadm_ids_table} addmissions_list 
        on addmissions_list.hadm_id = diagnoses_icd.hadm_id
    WHERE 
       (icd9_code like '870%' OR
        icd9_code like '871%' OR
        icd9_code like '872%' OR
        icd9_code like '873%' OR
        icd9_code like '874%' OR
        icd9_code like '875%' OR
        icd9_code like '876%' OR
        icd9_code like '877%' OR
        icd9_code like '878%' OR
        icd9_code like '879%' ) 
        """
    df = cursor.execute(query).as_pandas()
    return df
