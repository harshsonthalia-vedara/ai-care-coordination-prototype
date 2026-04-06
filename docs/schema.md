# Patient Data Schema

This project uses synthetic patient data for demonstration.

## Sections

### Patient
- member_id (string)
- first_name (string)
- last_name (string)
- age (int)
- sex (string)
- primary_conditions (list)
- risk_level (string)

### Claims
- date (string)
- diagnosis (string)

### Clinical Notes
- free-text notes

### Medications
- name
- adherence

### Labs
- test
- value
- interpretation

### Social Factors
- transportation_issue (boolean)
- food_access_issue (boolean)