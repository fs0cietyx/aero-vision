# Security Policy

## Supported Versions

AeroVision receives continuous security updates in accordance with the ISRO Cloud Removal Pipeline lifecycle. We strongly encourage utilizing the `main` branch to inherit the latest Deep Learning serialization fixes and CI/CD security parameters.

| Version | Supported          | Security Scope |
| ------- | ------------------ | -------------- |
| 1.0.x   | :white_check_mark: | Deep Learning deserialization & Archive Extraction |
| < 1.0   | :x:                | Unsupported |

## Reporting a Vulnerability

We take the security of geospatial data and inference pipelines extremely seriously. Given the potential application of this technology in government-grade remote sensing, any discovered vulnerability will be treated as critical.

If you discover a security vulnerability within AeroVision, please **DO NOT** open a public issue. Instead, follow these steps:

1. Send an email to the repository maintainer (via the contact info in our GitHub profile) outlining the exact vector of the vulnerability.
2. Include the Python version, PyTorch version, and OS where the bug was replicated.
3. Allow up to 48 hours for a direct technical response.

### What to Report?

*   **Insecure Deserialization Vectors:** Any bypass to our `weights_only=True` PyTorch load configuration.
*   **Arbitrary File Write / Path Traversal:** Any vectors allowing Zip Slip or zip bomb behavior bypassing our sanitization in `03_liss4_inference.py`.
*   **Supply Chain Weaknesses:** Found a vulnerable dependency not caught by Dependabot, Bandit, or CodeQL.

### Reward/Recognition
We will publicly acknowledge the reporter in our release notes and `SECURITY_ADVISORIES.md` file (unless you prefer to remain anonymous) after the vulnerability has been patched and merged.
