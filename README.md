# 🛡️ AI Credit Card Fraud Detection & Investigation Platform

> **Detect suspicious transactions. Explain the risk. Investigate with
> AI.**
>
> An end-to-end machine learning and agentic-AI project that combines
> fraud prediction, explainable AI, and a multi-agent investigation
> workflow to turn a transaction score into an understandable
> investigation report.

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-189AB4?style=for-the-badge&logo=xgboost&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white)
![LangGraph](https://img.shields.io/badge/Status-Phase%204%20complete-success)

------------------------------------------------------------------------

## 💡 Why this project?

A fraud detection model can tell us that a transaction looks suspicious.
But in a real investigation, a score alone is not enough.

An investigator also needs to understand:

-   How likely is this transaction to be fraudulent?
-   Which features contributed most to the prediction?
-   What does the model's evidence actually mean?
-   Which policy or investigation guidance may be relevant?
-   What should be reviewed next?

This project is being developed to connect those pieces into one
workflow: **machine learning predicts, SHAP explains, and AI agents help
organize the investigation evidence.**

## ✨ Key features

### 1. Machine-learning fraud prediction

-   Trains and uses an XGBoost classification model.
-   Produces a fraud probability for a transaction.
-   Applies a validation-derived decision threshold rather than assuming
    that 0.5 is always appropriate.
-   Returns a risk classification such as `HIGH_RISK` or `LOWER_RISK`.

### 2. Explainable AI with SHAP

-   Explains individual transaction predictions.
-   Identifies features that push a prediction toward or away from
    fraud.
-   Helps make model output more interpretable than a probability alone.
-   Keeps SHAP contribution values distinct from fraud probabilities.

### 3. AI-powered investigation assistance

-   Uses an OpenAI model through LangChain.
-   Builds an investigation narrative from supplied model predictions
    and SHAP evidence.
-   Uses explicit instructions to avoid inventing unavailable
    cardholder, merchant, location, or purchase details.
-   Distinguishes a model prediction from confirmed fraud.

### 4. Multi-agent workflow with LangGraph

The Phase 4 workflow coordinates four specialized nodes through shared
state:

  -----------------------------------------------------------------------
  Agent                               Responsibility
  ----------------------------------- -----------------------------------
  **Fraud Detection Agent**           Runs the trained model and
                                      generates SHAP evidence

  **Investigation Agent**             Produces an evidence-based AI
                                      investigation narrative

  **Knowledge Agent**                 Retrieves relevant policy guidance
                                      using the current initial
                                      keyword-based implementation

  **Reporting Agent**                 Creates a structured final
                                      investigation report
  -----------------------------------------------------------------------

Current workflow:

``` text
START
  |
  v
Fraud Detection Agent
  |
  v
Investigation Agent
  |
  v
Knowledge Agent
  |
  v
Reporting Agent
  |
  v
END
```

### 5. Structured investigation report

The workflow combines the model result, explanation, investigation
narrative, and retrieved policy context into a structured report that
can support human review.

### 6. Reproducible, phase-wise development

The project is organized into development phases and Git feature
branches. Each completed phase is committed and merged through a GitHub
Pull Request.

------------------------------------------------------------------------

## 🧠 How it works

At a high level, a transaction is passed into the workflow.

1.  **Load transaction data** --- prepare the feature values expected by
    the trained model.
2.  **Predict** --- XGBoost calculates the fraud probability and applies
    the selected threshold.
3.  **Explain** --- SHAP identifies important contributions for that
    prediction.
4.  **Investigate** --- the LLM receives the supplied prediction and
    explanation as evidence and creates a careful narrative.
5.  **Retrieve guidance** --- the Knowledge Agent supplies policy
    context. The current implementation uses a small illustrative policy
    knowledge base and keyword matching.
6.  **Report** --- the Reporting Agent combines the outputs into a
    structured result.

The workflow is designed to assist an investigator, not replace one.

## 🧰 Technology stack

  Area                          Technologies
  ----------------------------- ---------------------------------
  Language                      Python
  Data processing               Pandas, NumPy
  Machine learning              Scikit-learn, XGBoost
  Model explainability          SHAP
  Visualization / exploration   Matplotlib, Seaborn, JupyterLab
  LLM integration               OpenAI, LangChain
  Agent orchestration           LangGraph
  Model persistence             Joblib
  Environment configuration     python-dotenv
  Version control               Git, GitHub

> **Planned for Phase 5:** ChromaDB-based vector storage and semantic
> retrieval for the policy knowledge base. This is a planned
> enhancement, not part of the currently completed implementation.

------------------------------------------------------------------------

## 📁 Project structure

The structure below highlights the main project areas. Additional data,
model, notebook, and utility files may exist in the repository.

``` text
AI_Credit_Card_Fraud_Detection/
│
├── data/
│   ├── raw/                 # Original dataset (not committed)
│   └── processed/            # Processed data (not committed)
│
├── models/                   # Saved trained model artifacts (not committed)
│
├── notebooks/                # Exploratory data analysis and experiments
│
├── reports/
│   ├── figures/              # Generated plots
│   └── metrics/              # Evaluation outputs
│
├── src/
│   ├── agents/
│   │   ├── graph/
│   │   │   └── workflow.py
│   │   ├── nodes/
│   │   │   ├── fraud_detection_agent.py
│   │   │   ├── investigation_agent.py
│   │   │   ├── knowledge_agent.py
│   │   │   └── reporting_agent.py
│   │   ├── state/
│   │   │   └── fraud_state.py
│   │   └── fraud_investigation.py
│   │
│   ├── rag/
│   │   └── policy_knowledge.py
│   │
│   ├── data/
│   └── ml/
│
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## 🚀 Getting started

### Prerequisites

-   Python 3.11 recommended
-   Git
-   An OpenAI API key for the LLM-powered investigation component
-   The project dataset and trained model artifacts, when running the
    complete workflow

### 1. Clone the repository

``` bash
git clone https://github.com/vikzz25/AI_Credit_Card_Fraud_Detection.git
cd AI_Credit_Card_Fraud_Detection
```

### 2. Create and activate a virtual environment

**macOS / Linux**

``` bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows PowerShell**

``` powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

``` bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a local `.env` file in the project root. Do not commit this file.

``` env
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4.1-mini
```

Use your own valid API key. Keep `.env` private; `.env.example` should
contain placeholders only.

### 5. Prepare the data and model

The complete workflow expects the dataset and trained model artifact at
the paths configured by the project code. The raw dataset and generated
model files are intentionally excluded from Git to avoid committing
large or sensitive artifacts.

Run the project's data preparation and model training steps before
invoking the full investigation workflow. Refer to the source modules
and notebooks for the current pipeline entry points.

### 6. Run the investigation workflow

After the required dataset, trained model, and environment variables are
available, the current LangGraph workflow can be invoked with a
transaction represented as a feature dictionary.

Example:

``` python
from src.data.load_data import load_dataset
from src.agents.graph.workflow import build_fraud_investigation_graph

# Load the dataset and select a sample transaction.
df = load_dataset()
sample = df[df["Class"] == 1].iloc[0]

# Keep the label separate from the model input.
actual_label = int(sample["Class"])
transaction = sample.drop(labels=["Class"]).astype(float).to_dict()

# Build and execute the multi-agent workflow.
graph = build_fraud_investigation_graph()
result = graph.invoke({"transaction": transaction})

print("Actual label:", actual_label)
print("Prediction:", result["prediction"])
print("SHAP explanation:", result["shap_explanation"])
print("Investigation:", result["investigation"])
print("Policy context:", result["policy_context"])
print("Final report:", result["final_report"])
```

For a genuine transaction sample, change `df[df["Class"] == 1]` to
`df[df["Class"] == 0]`.

> This is an illustrative usage example based on the current project
> workflow. Exact output fields and setup requirements should be checked
> against the latest source code.

------------------------------------------------------------------------

## 📊 Dataset notes

The project has been developed using a credit-card transaction dataset
with anonymized features, including `Time`, `V1`--`V28`, transaction
`Amount`, and a binary `Class` label.

-   `Class = 0`: non-fraudulent transaction in the dataset.
-   `Class = 1`: fraudulent transaction in the dataset.
-   `V1`--`V28`: anonymized PCA-transformed features; their original
    business meanings are not available.
-   `Time`: elapsed time in seconds from the first transaction in the
    dataset, not a clock time.
-   `Amount`: transaction amount.

The dataset is highly imbalanced, so accuracy alone is not a sufficient
evaluation metric. Precision, recall, F1-score, confusion matrix, and
precision-recall analysis are important when evaluating fraud detection
performance.

Please follow the dataset provider's terms and licensing conditions when
obtaining or redistributing data.

## 🔐 Responsible AI and limitations

This project is an educational and engineering demonstration, not a
production banking fraud system.

-   A high model score is a **risk prediction**, not proof of fraud.
-   A low score does not guarantee that a transaction is legitimate.
-   SHAP values explain model behavior; they do not establish causation.
-   Anonymized PCA features cannot be translated into specific merchant,
    customer, or purchase explanations without additional data.
-   The current policy knowledge base is illustrative and is not
    official bank policy.
-   AI-generated narratives may be incomplete or incorrect and must be
    checked.
-   Human review and institution-approved procedures should guide
    consequential decisions. The system must not automatically block a
    card or reject a transaction solely because of an AI-generated
    explanation.

Before any real-world deployment, the project would require rigorous
validation, security and privacy review, monitoring, auditability,
governance, and integration with approved operational procedures.

## 🗺️ Development roadmap

The project is being built incrementally:

-   [x] **Phase 1 --- Baseline machine learning:** data exploration and
    initial fraud classification pipeline.
-   [x] **Phase 2 --- XGBoost and SHAP:** improved model development,
    evaluation, thresholding, and explainability.
-   [x] **Phase 3 --- OpenAI and LangChain:** AI-assisted transaction
    investigation.
-   [x] **Phase 4 --- LangGraph multi-agent workflow:** coordinated
    detection, investigation, knowledge, and reporting agents.
-   [ ] **Phase 5 --- RAG with ChromaDB:** document ingestion, chunking,
    embeddings, vector storage, and semantic policy retrieval.
-   [ ] **Phase 6 --- Backend API:** expose the fraud investigation
    workflow through an API.
-   [ ] **Phase 7 --- Dashboard:** build an interface for transaction
    review and investigation reports.
-   [ ] **Phase 8 --- Deployment:** package and deploy the application.

The roadmap reflects project intent; unchecked phases are planned and
should not be considered implemented.

## 🌿 Git workflow

Development is organized by feature branch and phase. The completed
Phase 4 branch was merged into `main`.

Example branch naming:

``` text
feature/phase-1-baseline-ml
feature/phase-2-xgboost-shap
feature/phase-3-openai-langchain
feature/phase-4-langgraph-agents
feature/phase-5-rag-chromadb
feature/phase-6-backend-api
feature/phase-7-dashboard
feature/phase-8-deployment
```

A typical contribution flow:

``` bash
git checkout main
git pull origin main
git checkout -b feature/your-feature
# Make and test your changes
git add <files>
git commit -m "feat: describe your change"
git push -u origin feature/your-feature
```

Open a Pull Request against `main` and merge after review.

## 🤝 Contributions

Suggestions, bug reports, and improvements are welcome. If you would
like to contribute:

1.  Fork the repository.
2.  Create a focused feature branch.
3.  Make a small, testable change.
4.  Run relevant checks.
5.  Open a Pull Request describing the change and how it was tested.

## 👨‍💻 Author

**Vikas Bankar**

GitHub: [@vikzz25](https://github.com/vikzz25)

Project: [AI Credit Card Fraud
Detection](https://github.com/vikzz25/AI_Credit_Card_Fraud_Detection)

------------------------------------------------------------------------

```{=html}
<p align="center">
```
`<b>`{=html}From a fraud score to an explainable investigation --- one
phase at a time. 🚀`</b>`{=html}
```{=html}
</p>
```
