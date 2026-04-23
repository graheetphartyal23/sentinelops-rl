"""Catalog of large datasets used for SentinelOps task generation."""

from __future__ import annotations

CORPUS_SOURCES = {
    "bgl_logs": {
        "description": "Blue Gene/L system logs for anomaly and incident sequence modeling.",
        "url": "https://github.com/logpai/loghub/tree/master/BGL",
        "target_rows": 4000000,
    },
    "hdfs_logs": {
        "description": "HDFS log anomaly corpus for infrastructure event realism.",
        "url": "https://github.com/logpai/loghub/tree/master/HDFS",
        "target_rows": 11000000,
    },
    "nvd_cve": {
        "description": "National Vulnerability Database CVE feed for remediation planning.",
        "url": "https://nvd.nist.gov/vuln/data-feeds",
        "target_rows": 250000,
    },
    "cisa_kev": {
        "description": "CISA Known Exploited Vulnerabilities catalog for risk prioritization.",
        "url": "https://www.cisa.gov/known-exploited-vulnerabilities-catalog",
        "target_rows": 1000,
    },
    "mitre_attack": {
        "description": "MITRE ATT&CK techniques for tactic and procedure mapping.",
        "url": "https://attack.mitre.org/",
        "target_rows": 10000,
    },
    "enron_email": {
        "description": "Enron email corpus for stakeholder communication style modeling.",
        "url": "https://www.cs.cmu.edu/~enron/",
        "target_rows": 500000,
    },
}
