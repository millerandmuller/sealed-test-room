# Sealed Test Room

**A scientist gets to try a new, untrusted analysis tool on confidential study data, and IT gets a receipt that proves the tool could reach exactly one host, and that the room is gone.**

Built for the Vultr Agent Rush hackathon (Challenge 1, "Blast Radius Zero").

![A receipt from a real run: the connection table (two hosts refused, one allowed) and the CLEAN banner, with the synthetic-data badge](docs/receipt.png)

*A real receipt from a run on a Vultr VX1 host. The data in it is synthetic and says so. The connection table is the point: the package index and the Git host are refused from inside the room, the inference endpoint is reachable, and the room is proven gone at the end.*

## Who it is for

- **The study director (the scientist).** She has a new prediction tool and a confidential study file, and change control says "not yet" to anything unreviewed. She hands the room the tool and the file and gets results back.
- **The approver (IT or QA).** One page, no trust in a model needed: which tool and version, the exact code that ran, hashes of every file in and out, which connections were attempted and what happened to each, and when the room existed and the proof it is gone.
- **The buyer (the manager of a regulated-data team).** Pharma safety groups, contract research organizations and consultancies that keep saying "not yet" to useful tools because nobody can show what a tool did.

**Pricing hypothesis (a hypothesis, not a measured number):** per sealed run, because the unit the approver signs off is one run with one receipt.

## Why the room has one door

A tool that calls a model needs exactly one host. So the sealed room is created with no network except one address, the Vultr Serverless Inference endpoint, and the **wall check** proves that before any study data enters: from inside the room the package index and the Git host must be refused and the endpoint must answer. Anything else stops the run.

In this version the room uses its door once, near the end: a short result check is made **from inside the room** to the endpoint (the receipt shows it as "Calls through the door: N", with host, latency and model). What happens to the inference key, exactly:

> The agent loop runs on the host. The inference key is never in the room while the tool runs. For the single result-verification call it is passed on the standard input of that one command (not in the environment, not on the command line, not written to the room's disk), the room can reach only the inference endpoint, and the room is destroyed afterwards. The call is reported by code running in the room: if a tool had tampered with the room's own Python installation, that report could be wrong, so the wall check and the host's own result check do not depend on it.

This is the narrowest route that worked. Microsandbox 0.7.13 has a host-scoped secret option (`msb create --secret`), but it is refused by the proxy when the room's policy is an IP-only rule, and the variant that works needs open DNS and TLS interception, which would weaken the wall. Both were measured on the host (the first answers `403 secret_policy_denied`); `scripts/door_probe.py` re-checks the shipped route and prints only MATCH or NO MATCH for the key in the room's environment and on its disk.

The host's own check of the results stays authoritative; the model's reply is recorded next to it, never trusted over it.

## Scope

v1 supports one approved tool (`admet-ai` 2.0.1); the policy (one door, wall check, destroy check, receipt) applies to any tool, and the agent loop is tested on this one. Any other tool name posted to the app is refused before a run or a room exists.

## How a run works

![Architecture: web app and orchestrator on a Vultr VX1 host; a data-free download room; a sealed room with one door to Vultr Serverless Inference; a receipt](docs/architecture.svg)

1. **Download room** (a Microsandbox microVM, default network, no study data inside): the tool's wheels are fetched, hashed, packed into one tar file and copied out. The room is destroyed.
2. **Sealed room** (a microVM created with `--no-net` and one allow rule for the inference endpoint's IP on port 443): the wheels are copied in and installed offline.
3. **Wall check**, before any data enters: from inside the sealed room the agent tries the package index, the Git host and the inference endpoint. Two must be refused and one must succeed.
4. **Data in, results out**: the study file is copied in (SHA-256 recorded), the agent writes and runs a runner script, the results file is copied out (SHA-256 recorded). The host checks the results itself.
5. **The door**: the room makes its one result-check call to the endpoint (see above).
6. **Destroy check**: the room is removed and `msb list` is run to prove it is gone.
7. **Receipt**: one JSON record per run (schema in `schemas/receipt.schema.json`) and one printable HTML page.

## Vultr services used

- **Vultr Cloud Compute, VX1 (`vx1-g-2c-8g-120s`, New York)**: hosts the web app, the orchestrator and the microVMs; `/dev/kvm` is required.
- **Microsandbox (`msb` 0.7.13) on that host**: creates, executes in, copies to and from, and removes the rooms.
- **Vultr Serverless Inference** (OpenAI-compatible, `glm-5.3`, fallback `qwen3.8-27b`): runs the agent loop, makes the in-room result check, and is the single host the sealed room may reach.

## Try it

The app is served at **https://45-63-20-61.sslip.io**. It asks for an access code once (a signed cookie keeps it for the browser session); the code for judges is given in the submission text and is never stored in this repository. A wrong code starts nothing: no run, no room. The code is set by `JUDGE_ACCESS_CODE` in the host's environment file.

The site is served over HTTPS (Caddy with a Let's Encrypt certificate, on a hostname derived from the host's IP address through the sslip.io wildcard DNS service). Some networks refuse to resolve sslip.io names; for those, `http://45.63.20.61` serves the same app over plain HTTP, where the access code is not encrypted in transit.

Choose the synthetic demo study and press "Run in sealed room". The wall-check rows appear after about 20 seconds; a full run takes a few minutes. Only one run can be in progress at a time.

To run it yourself: Linux host with `/dev/kvm`, `msb` 0.7.13, Python 3.12, `pip install -r requirements.txt`, set `VULTR_INFERENCE_API_KEY` and `JUDGE_ACCESS_CODE`, then `uvicorn sealed_room.web:create_app --factory`. Tests: `pip install -r requirements-dev.txt && pytest`.

## What this is not

- The demo study data is synthetic and says so on every screen.
- A run is not a validated or GLP-compliant process, and nothing here claims approval by any real IT department. The "approved list" in the app is the demo's own list.
- The tool's numbers are the tool's output as measured in that run; they are not a claim that the tool is accurate.
- Domain names in Microsandbox 0.7.13 allow rules break TLS, so the policy uses the endpoint's IP address, resolved on the host at room creation and recorded in the receipt.
- Tool output the agent reads is sent to the inference endpoint, which states that data is processed in memory and not stored persistently by default (docs.vultr.com, 2026-10-26).

## License

MIT, see `LICENSE`.
