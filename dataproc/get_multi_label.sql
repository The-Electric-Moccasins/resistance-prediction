SELECT
   hadm_id,
   array_agg(ARRAY[org_itemid, ab_name, RESISTANT_YN])
from
(
SELECT
    microb.hadm_id,
    microb.ab_name,
    cast(microb.org_itemid as varchar) org_itemid,
    CASE WHEN microb.interpretation in ('R','I') THEN '1' ELSE '0' END AS RESISTANT_YN

FROM mimiciii.admissions admits
INNER JOIN mimiciii.microbiologyevents microb
    ON microb.hadm_id = admits.hadm_id
WHERE ab_name in
  ('CEFTAZIDIME')
  -- ('CEFAZOLIN', 'CEFEPIME', 'CEFPODOXIME', 'CEFTAZIDIME', 'CEFTRIAXONE', 'CEFUROXIME')

    AND
      org_itemid in (
                80004, -- KLEBSIELLA PNEUMONIAE
                80026, -- PSEUDOMONAS AERUGINOSA
                80005, -- KLEBSIELLA OXYTOCA
                80017, -- PROTEUS MIRABILIS
                80040, -- NEISSERIA GONORRHOEAE
                80008, -- ENTEROBACTER CLOACAE
                80007, -- ENTEROBACTER AEROGENES
                80002) -- ESCHERICHIA COLI

) mb
group by 1
order by hadm_id
