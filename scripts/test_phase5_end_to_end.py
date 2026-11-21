
"""
Phase 5: End-to-End Fraud Investigation Test.

Runs all four LangGraph agents with CPU thread limits
to avoid native CPU thread synchronization issues on macOS.
"""

# ---------------------------------------------------------
# 1. CPU CONFIGURATION
# ---------------------------------------------------------

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import PyTorch only after setting environment variables.
import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

print("PyTorch CPU threads:", torch.get_num_threads(), flush=True)


# ---------------------------------------------------------
# 2. CONFIGURE PROJECT PATH AND IMPORT MODULES
# ---------------------------------------------------------

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.fraud_investigation import load_fraud_model
from src.data.load_data import load_dataset
from src.agents.graph.workflow import build_fraud_investigation_graph


# ---------------------------------------------------------
# 3. LOAD MODEL AND DATASET
# ---------------------------------------------------------

print("\n" + "=" * 70, flush=True)
print("PHASE 5 - END-TO-END FRAUD INVESTIGATION", flush=True)
print("=" * 70, flush=True)

print("\nStep 1: Loading fraud detection model...", flush=True)

model_bundle = load_fraud_model()

print("Model loaded successfully.", flush=True)


print("\nStep 2: Loading dataset...", flush=True)

df = load_dataset()

print("Dataset loaded successfully.", flush=True)
print("Dataset shape:", df.shape, flush=True)


# ---------------------------------------------------------
# 4. PREPARE TRANSACTION
# ---------------------------------------------------------

print("\nStep 3: Preparing transaction...", flush=True)

transaction = (
    df[model_bundle["feature_columns"]]
    .iloc[0]
    .to_dict()
)

print("Transaction prepared successfully.", flush=True)
print("Number of features:", len(transaction), flush=True)


# ---------------------------------------------------------
# 5. BUILD LANGGRAPH
# ---------------------------------------------------------

print("\nStep 4: Building LangGraph workflow...", flush=True)

graph = build_fraud_investigation_graph()

print("Workflow compiled successfully.", flush=True)

print("\nWORKFLOW STRUCTURE", flush=True)
print("START -> fraud_detection -> investigation -> knowledge -> reporting -> END", flush=True)


# ---------------------------------------------------------
# 6. EXECUTE COMPLETE WORKFLOW
# ---------------------------------------------------------

print("\nStep 5: Executing complete investigation...", flush=True)
print("The Investigation Agent will make an OpenAI API call.", flush=True)

result = graph.invoke({
    "transaction": transaction
})

print("\nComplete investigation finished!", flush=True)


# ---------------------------------------------------------
# 7. VALIDATE RESULTS
# ---------------------------------------------------------

print("\n" + "=" * 70, flush=True)
print("PHASE 5 VALIDATION", flush=True)
print("=" * 70, flush=True)

checks = {
    "Fraud prediction generated": bool(result.get("prediction")),
    "SHAP explanation generated": bool(result.get("shap_explanation")),
    "Investigation generated": bool(result.get("investigation")),
    "Policy context retrieved": bool(result.get("policy_context")),
    "Final report generated": bool(result.get("final_report")),
    "Workflow completed": result.get("status") == "REPORT_COMPLETED",
}

for description, passed in checks.items():
    print(
        f"[{'PASS' if passed else 'FAIL'}] {description}",
        flush=True
    )


# ---------------------------------------------------------
# 8. DISPLAY FINAL REPORT
# ---------------------------------------------------------

final_report = result.get("final_report", {})

if final_report:
    print("\nFINAL FRAUD INVESTIGATION REPORT", flush=True)
    print("=" * 70, flush=True)

    for section, content in final_report.items():
        print(f"\n{section.upper()}", flush=True)
        print("-" * 50, flush=True)
        print(content, flush=True)
else:
    print("No final report was returned.", flush=True)


# ---------------------------------------------------------
# 9. FINAL RESULT
# ---------------------------------------------------------

print("\n" + "=" * 70, flush=True)

if all(checks.values()):
    print("PHASE 5 END-TO-END TEST PASSED!", flush=True)
else:
    print("PHASE 5 END-TO-END TEST NEEDS INVESTIGATION.", flush=True)

print("=" * 70, flush=True)
