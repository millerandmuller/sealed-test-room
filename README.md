# Sealed Test Room

A scientist hands the room a new, untrusted analysis tool and her confidential study data. An AI agent sets the tool up and runs it on a Vultr machine that can reach exactly one host (the Vultr inference endpoint), returns the results and a receipt IT can read, and the room is destroyed.

Status: work in progress, built for the Vultr Agent Rush hackathon (Challenge 1, "Blast Radius Zero").

## How a run works

1. **Download room** (a Microsandbox microVM, no study data inside): the tool's wheels are fetched, hashed, packed into one tar file and copied out. The room is destroyed.
2. **Sealed room** (a microVM created with `--no-net` and one allow rule for the inference endpoint IP on port 443): the wheels are copied in and installed offline.
3. **Wall check**, before any data enters: from inside the sealed room the agent tries the package index, the Git host and the inference endpoint. Two must be refused and one must succeed. Anything else stops the run.
4. **Data in, results out**: the study file is copied in (SHA-256 recorded), the agent writes and runs a runner script, the results file is copied out (SHA-256 recorded).
5. **Destroy check**: the room is removed and `msb list` is run to prove it is gone.
6. **Receipt**: one JSON record per run (schema in `schemas/receipt.schema.json`) and one printable HTML page.

## Vultr services used

- **Vultr Cloud Compute, VX1 (`vx1-g-2c-8g-120s`, New York)**: hosts the web app, the orchestrator and the microVMs; `/dev/kvm` is required.
- **Microsandbox (`msb` 0.7.13) on that host**: creates, executes in, copies to and from, and removes the rooms.
- **Vultr Serverless Inference** (OpenAI-compatible, `glm-5.3`, fallback `qwen3.8-27b`): runs the agent loop; it is also the single host the sealed room may reach.

## Access

The public URL asks for an access code once (a signed cookie keeps it for the browser session). A wrong code starts nothing: no run, no room. The code is set by `JUDGE_ACCESS_CODE` in the host's environment file and is never stored in the repository. The site is served over plain HTTP, so the code is not encrypted in transit.

## What this is not

- The demo study data is synthetic and says so on every screen.
- A run is not a validated or GLP-compliant process, and nothing here claims IT approval.
- Domain names in Microsandbox 0.7.13 allow rules break TLS, so the policy uses the endpoint's IP address, resolved on the host at room creation and recorded in the receipt.

## License

MIT, see `LICENSE`.
