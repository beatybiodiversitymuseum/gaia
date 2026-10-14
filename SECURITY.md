# Security Policy

## Reporting a vulnerability

Do not report suspected vulnerabilities in a public issue. Use GitHub's private
vulnerability-reporting feature when it is enabled for this repository, or
contact the repository maintainers through an established private channel.

Include the affected version or commit, reproduction steps, impact, and any
suggested mitigation. Avoid including real credentials, private specimen data,
or other sensitive material in the report.

## Supported versions

Until Gaia publishes versioned releases, security fixes are made on the latest
commit of the default branch only.

## Deployment boundary

Gaia's Python HTTP bridge is intended to remain on a trusted loopback or private
network. Public deployments must place an authenticated frontend or reverse
proxy in front of it, configure independent UI and session secrets, and avoid
exposing PostgreSQL or the model server directly.
