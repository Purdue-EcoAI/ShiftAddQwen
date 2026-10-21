"""Display names: run labels -> names used in the README, tables and figures."""
BASELINE_REPO = "unsloth/Qwen3.5-9B-MTP-GGUF"


def display(label):
    """'v2g4.0_ft' -> 'ShiftAddQwen-3.5-9B-4.0bits'; 'UD-IQ3_XXS' -> 'UnslothAI/Qwen-3.5-9B-UD-IQ3_XXS'."""
    if label.startswith("v2g"):
        return f"ShiftAddQwen-3.5-9B-{label[3:].replace('_ft', '').replace('q6', '')}bits"
    if label.startswith(("k", "sa3")):
        return "ShiftAddQwen-ablation-" + label
    return "UnslothAI/Qwen-3.5-9B-" + label


def short(label):
    """figure point labels: '4.0 bits' / 'Q4_K_M'"""
    if label.startswith("v2g"):
        return label[3:].replace("_ft", "").replace("q6", "") + " bits"
    return label


def is_ours(label):
    return label.startswith("v2g")
