SELECT
    dem."customerID", dem."gender", dem."SeniorCitizen", dem."Partner", dem."Dependents",
    ser."PhoneService", ser."MultipleLines", ser."InternetService", ser."OnlineSecurity", ser."OnlineBackup", ser."DeviceProtection",
    acc."tenure", acc."Contract", acc."PaperlessBilling", acc."PaymentMethod", acc."MonthlyCharges", acc."TotalCharges", acc."Churn"
FROM customers_demographic dem
LEFT JOIN customers_account_info acc ON dem."customerID" = acc."customerID"
LEFT JOIN customers_services ser ON dem."customerID" = ser."customerID"