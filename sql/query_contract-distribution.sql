WITH
    contract_general AS (
        SELECT "Contract", COUNT(1) AS "Frequency"
        FROM customers_account_info
        GROUP BY "Contract"
    ),
    contract_churn_no AS (
        SELECT "Contract", COUNT(1) AS "Frequency"
        FROM customers_account_info
        WHERE "Churn" = 'No'
        GROUP BY "Contract"
    ),
    contract_churn_yes AS (
        SELECT "Contract", COUNT(1) AS "Frequency"
        FROM customers_account_info
        WHERE "Churn" = 'Yes'
        GROUP BY "Contract"
)

SELECT
    gen."Contract",
    gen."Frequency" AS "Freq_General",
    n."Frequency" AS "Freq_Churn-No",
    y."Frequency" AS "Freq_Churn-Yes",
    n."Frequency" * 100.0 / gen."Frequency" AS "Prop_Churn-No",
    y."Frequency" * 100.0 / gen."Frequency" AS "Prop_Churn-Year"
FROM contract_general gen
LEFT JOIN contract_churn_no n
    ON gen."Contract" = n."Contract"
LEFT JOIN contract_churn_yes y
    ON gen."Contract" = y."Contract"