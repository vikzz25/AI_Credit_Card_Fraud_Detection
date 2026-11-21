# Credit Card Fraud Detection Policy

## 1. Purpose

This document defines general guidelines for identifying, assessing, and investigating potentially fraudulent credit card transactions.

The objective is to support fraud analysts in identifying suspicious transaction patterns and making informed investigation decisions.

## 2. Common Fraud Indicators

The following indicators may suggest potentially suspicious activity:

- Unusually large transactions compared with historical spending patterns.
- Multiple transactions within a very short period.
- Transactions originating from unusual geographic locations.
- Repeated failed transaction attempts.
- Multiple transactions at merchants associated with suspicious activity.
- Sudden changes in transaction frequency or spending behavior.

A single indicator does not establish that a transaction is fraudulent.

## 3. Risk Classification

### LOW RISK

Characteristics:
- Transaction behavior appears consistent with normal activity.
- No significant suspicious indicators are identified.
- Model-generated fraud probability is relatively low.

Recommended action:
- Allow normal monitoring.
- Retain transaction records for auditing.

### MEDIUM RISK

Characteristics:
- One or more unusual transaction patterns are identified.
- Transaction behavior differs from historical patterns.
- Additional verification may be appropriate.

Recommended action:
- Perform additional investigation.
- Review available transaction evidence.
- Consider customer verification.

### HIGH RISK

Characteristics:
- Multiple suspicious transaction indicators are identified.
- Model-generated fraud probability is elevated.
- Supporting evidence suggests potentially fraudulent activity.

Recommended action:
- Prioritize investigation.
- Review transaction history and supporting evidence.
- Escalate to a human fraud analyst.

## 4. Investigation Guidelines

Fraud investigators should:

1. Review the transaction details.
2. Examine the model-generated fraud probability.
3. Review feature importance and SHAP explanations.
4. Identify unusual transaction characteristics.
5. Compare available evidence against fraud indicators.
6. Document findings and recommendations.
7. Escalate suspicious cases for human review when appropriate.

## 5. Important Considerations

- Machine learning predictions are not proof of fraud.
- SHAP explanations describe model behavior, not necessarily real-world causation.
- An unusual transaction is not automatically fraudulent.
- Missing information should not be treated as evidence of fraud.
- Final decisions must involve authorized human reviewers.
- No automatic card blocking or financial action should be performed solely based on AI-generated recommendations.

## 6. Human Review

All high-risk cases should be prioritized for review by an authorized fraud analyst.

The investigator should consider the complete available evidence before making a final decision.

---

Document Type: Fraud Detection Policy
Document Version: 1.0
Purpose: AI Fraud Investigation Knowledge Base
Status: Illustrative educational policy