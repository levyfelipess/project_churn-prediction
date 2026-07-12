SELECT "Churn", COUNT(1) AS "Frequency"
FROM customers_account_info
GROUP BY "Churn"