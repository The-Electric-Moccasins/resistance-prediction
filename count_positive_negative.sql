-- count positives, negative

select label, count(*) from (
select hadm_id, max(RESISTANT_YN) label from (
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
    SENSITIVE_YN
FROM admissions
WHERE admissions.time_to_rslt is not null
      and admissions.time_to_rslt >= 24

) a
group by 1
) b
group by 1