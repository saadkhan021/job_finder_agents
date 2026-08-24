from __future__ import annotations

# Heuristic only: there is no reliable "employee count" field on scraped
# LinkedIn/Indeed job cards, so this classifies by matching well-known
# large/multinational company names. Anything that doesn't match is
# bucketed as "startup" (i.e. smaller/local company) by default.
#
# This is a best-effort filter, not exact data -- a local company not on
# this list will be treated as a "startup", and it won't catch every
# multinational either. Extend the list below as needed.

MULTINATIONAL_KEYWORDS = [
    "google", "microsoft", "amazon", "meta", "facebook", "apple", "ibm",
    "oracle", "sap", "accenture", "deloitte", "pwc", "ey ", "ernst & young",
    "kpmg", "unilever", "nestle", "procter & gamble", "p&g", "coca-cola",
    "pepsico", "shell", "totalenergies", "siemens", "philips", "samsung",
    "huawei", "tcs", "infosys", "wipro", "cognizant", "telenor", "jazz",
    "ufone", "zong", "habib bank", "hbl", "united bank", "ubl",
    "standard chartered", "citibank", "jp morgan", "hsbc", "abbott",
    "gsk", "glaxosmithkline", "bayer", "novartis", "pfizer",
    "johnson & johnson", "toyota", "honda", "suzuki", "nissan",
    "engro", "systems limited", "netsol", "byco", "fauji fertilizer",
    "packages limited", "lucky cement", "servicenow", "salesforce",
    "adobe", "intel", "cisco", "dell", "hp inc", "hewlett packard",
]


def classify_company(company_name: str) -> str:
    """
    Return "multinational" if the company name matches a known large/
    multinational brand, otherwise "startup" (treated as smaller/local).
    """

    if not company_name:
        return "startup"

    normalized = company_name.strip().lower()

    for keyword in MULTINATIONAL_KEYWORDS:
        if keyword in normalized:
            return "multinational"

    return "startup"