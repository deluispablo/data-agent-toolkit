# Security Policy

## Supported versions

Each agent is versioned independently. Security fixes land on `main` and
ship in the next release of the affected agent; while an agent is `0.x`,
only its latest release is supported.

| Agent | Supported |
|---|---|
| `csv-inspector` | latest `0.x` release |

## Reporting a vulnerability

**Do not open a public issue.** Report it privately through GitHub's
[private vulnerability reporting](https://github.com/deluispablo/data-agent-toolkit/security/advisories/new)
("Report a vulnerability" in the repository's **Security** tab).

Please include the affected agent and version, a description of the issue
and its impact, and the steps or a minimal input to reproduce it. This is a
single-maintainer project: the aim is to acknowledge a report within a week
and keep you updated until it is resolved. Once a fix is released, the
advisory is published with credit to the reporter, unless you prefer
otherwise.

## Scope

Of particular interest, given what the agents do:

- **Credential leaks**: an API key or other secret appearing in logs,
  exception messages, `repr` output or tracebacks.
- **Unbounded resource use from untrusted input**: a CSV source that makes
  `csv-inspector` read or hold more than its documented byte budgets, or
  run past its time budget.
- **Implicit configuration loading**: the library reading a `.env` file or
  other configuration the host did not ask for.

The accuracy of an LLM's answer is not a security issue; report wrong
inspection results as a regular bug.

The hosts under `examples/` are out of scope. They are executable
documentation, never released or shipped, so a weakness in an example's
own code is a regular bug. A vulnerability in an agent that an example
merely makes visible is in scope, reported against the agent.
