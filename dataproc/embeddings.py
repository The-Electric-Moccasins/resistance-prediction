# Import libraries
from pyathena import connect
from pyathena.pandas.cursor import PandasCursor
from pyathena.pandas.util import as_pandas
import boto3
from botocore.client import ClientError
import pandas as pd

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

def loinc_values(lab_item_ids) -> pd.DataFrame:
    statement = """
    SELECT 
        lab_events.itemid lab_item_id, 
        lab_events.value
    FROM mimiciii.labevents lab_events
    WHERE B.itemid IN ({});
    """.format(str(lab_item_ids)[1:-1])
    df = cursor.execute(statement).as_pandas()
    return df