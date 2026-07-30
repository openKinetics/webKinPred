/**
 * ApiDocs — the /api-docs documentation page.
 *
 * Structured as a single scrollable page with the following sections:
 *   1. Hero header with base URL
 *   2. Quick start (4-step overview)
 *   3. Authentication
 *   4. Endpoint reference (collapsible cards, each with request + response examples)
 *   5. Full code examples (Python / curl tabs)
 *   6. CSV format reference
 *   7. Rate limits
 *   8. Error reference
 */

import React from 'react';
import './ApiDocs.css';
import CodeBlock from './CodeBlock';
import EndpointCard from './EndpointCard';
import TabbedCode from './TabbedCode';
import ApiKeyManager from './ApiKeyManager';

const BASE = 'https://predictor.openkinetics.org/api/v1';

// ---------------------------------------------------------------------------
// Per-endpoint request examples
// ---------------------------------------------------------------------------

const HEALTH_REQ = [
  {
    label: 'Python',
    code:
`import requests

resp = requests.get("${BASE}/health/")
print(resp.json())`,
  },
  {
    label: 'curl',
    code: `curl ${BASE}/health/`,
  },
];

const METHODS_REQ = [
  {
    label: 'Python',
    code:
`import requests

resp = requests.get("${BASE}/methods/")
print(resp.json())`,
  },
  {
    label: 'curl',
    code: `curl ${BASE}/methods/`,
  },
];

const QUOTA_REQ = [
  {
    label: 'Python',
    code:
`import requests

resp = requests.get(
    "${BASE}/quota/",
    headers={"Authorization": "Bearer ak_your_key_here"},
)
print(resp.json())`,
  },
  {
    label: 'curl',
    code:
`curl "${BASE}/quota/" \\
  -H "Authorization: Bearer ak_your_key_here"`,
  },
];

const VALIDATE_REQ = [
  {
    label: 'Python',
    code:
`import requests

API_KEY = "ak_your_key_here"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

# Validate without similarity (fast)
with open("input.csv", "rb") as f:
    resp = requests.post(
        "${BASE}/validate/",
        headers=HEADERS,
        files={"file": f},
        data={"runSimilarity": "false"},
    )
print(resp.json())

# Validate WITH similarity (blocks until MMseqs2 finishes)
with open("input.csv", "rb") as f:
    resp = requests.post(
        "${BASE}/validate/",
        headers=HEADERS,
        files={"file": f},
        data={"runSimilarity": "true"},
        timeout=600,
    )
print(resp.json())`,
  },
  {
    label: 'curl',
    code:
`# Validate without similarity
curl -X POST "${BASE}/validate/" \\
  -H "Authorization: Bearer ak_your_key_here" \\
  -F "file=@input.csv" \\
  -F "runSimilarity=false"

# Validate WITH similarity
curl -X POST "${BASE}/validate/" \\
  -H "Authorization: Bearer ak_your_key_here" \\
  -F "file=@input.csv" \\
  -F "runSimilarity=true"`,
  },
];

const SUBMIT_REQ = [
  {
    label: 'Python',
    code:
`import requests

API_KEY = "ak_your_key_here"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

with open("input.csv", "rb") as f:
    resp = requests.post(
        "${BASE}/submit/",
        headers=HEADERS,
        files={"file": f},
        data={
            "targets":             '["kcat"]',
            "methods":             '{"kcat":"DLKcat"}',
            "handleLongSequences": "truncate",
            "includeSimilarityColumns": "true",
            "canonicalizeSubstrates": "true",
        },
    )
print(resp.json())`,
  },
  {
    label: 'curl',
    code:
`curl -X POST "${BASE}/submit/" \\
  -H "Authorization: Bearer ak_your_key_here" \\
  -F "file=@input.csv" \\
  -F 'targets=["kcat"]' \\
  -F 'methods={"kcat":"DLKcat"}' \\
  -F "handleLongSequences=truncate" \\
  -F "includeSimilarityColumns=true" \\
  -F "canonicalizeSubstrates=true"`,
  },
];

const STATUS_REQ = [
  {
    label: 'Python',
    code:
`import requests

resp = requests.get(
    "${BASE}/status/aB3kX9z/",
    headers={"Authorization": "Bearer ak_your_key_here"},
)
print(resp.json())`,
  },
  {
    label: 'curl',
    code:
`curl "${BASE}/status/aB3kX9z/" \\
  -H "Authorization: Bearer ak_your_key_here"`,
  },
];

const RESULT_REQ = [
  {
    label: 'Python',
    code:
`import requests

HEADERS = {"Authorization": "Bearer ak_your_key_here"}

# Download as CSV
resp = requests.get("${BASE}/result/aB3kX9z/", headers=HEADERS)
with open("output.csv", "wb") as f:
    f.write(resp.content)

# Or receive JSON directly
resp_json = requests.get(
    "${BASE}/result/aB3kX9z/?format=json",
    headers=HEADERS,
)
print(resp_json.json())`,
  },
  {
    label: 'curl',
    code:
`# Download as CSV
curl "${BASE}/result/aB3kX9z/" \\
  -H "Authorization: Bearer ak_your_key_here" \\
  -o output.csv

# Or get JSON
curl "${BASE}/result/aB3kX9z/?format=json" \\
  -H "Authorization: Bearer ak_your_key_here"`,
  },
];

// ---------------------------------------------------------------------------
// Full workflow examples (Python + curl only)
// ---------------------------------------------------------------------------

const PYTHON_FULL = `import requests
import time

API_KEY = "ak_your_key_here"
BASE    = "${BASE}"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

# ── Step 1: Submit a job ────────────────────────────────────────────────────
with open("input.csv", "rb") as f:
    response = requests.post(
        f"{BASE}/submit/",
        headers=HEADERS,
        files={"file": f},
        data={
            "targets":             '["kcat"]',
            "methods":             '{"kcat":"DLKcat"}',
            "handleLongSequences": "truncate",
            "useExperimental":     "true",
            "includeSimilarityColumns": "true",
            "canonicalizeSubstrates": "true",
        },
    )
response.raise_for_status()
job = response.json()

print(f"Job submitted  → {job['jobId']}")
print(f"Quota remaining: {job['quota']['remaining']:,} predictions")

# ── Step 2: Poll until complete ─────────────────────────────────────────────
while True:
    status_resp = requests.get(f"{BASE}/status/{job['jobId']}/", headers=HEADERS)
    status_resp.raise_for_status()
    status = status_resp.json()

    print(f"  Status: {status['status']}  ({status['elapsedSeconds']}s elapsed)")

    if status["status"] == "Completed":
        break
    if status["status"] == "Failed":
        print(f"  Error: {status.get('error')}")
        raise SystemExit(1)

    time.sleep(5)

# ── Step 3: Download results ─────────────────────────────────────────────────
result = requests.get(f"{BASE}/result/{job['jobId']}/", headers=HEADERS)
result.raise_for_status()

with open("output.csv", "wb") as f:
    f.write(result.content)

print("Results saved to output.csv")`;

const CURL_FULL = `# Set your credentials
API_KEY="ak_your_key_here"
BASE="${BASE}"

# ── Step 1: Submit a job ────────────────────────────────────────────────────
JOB=$(curl -s -X POST "$BASE/submit/" \\
  -H "Authorization: Bearer $API_KEY" \\
  -F "file=@input.csv" \\
  -F 'targets=["kcat"]' \\
  -F 'methods={"kcat":"DLKcat"}' \\
  -F "handleLongSequences=truncate" \\
  -F "useExperimental=true" \\
  -F "includeSimilarityColumns=true" \\
  -F "canonicalizeSubstrates=true")

JOB_ID=$(echo "$JOB" | python3 -c "import sys,json; print(json.load(sys.stdin)['jobId'])")
echo "Job submitted → $JOB_ID"

# ── Step 2: Poll until complete ─────────────────────────────────────────────
while true; do
  STATUS=$(curl -s "$BASE/status/$JOB_ID/" -H "Authorization: Bearer $API_KEY")
  STATE=$(echo "$STATUS" | python3 -c "import sys,json; print(json.load(sys.stdin)['status'])")
  echo "  Status: $STATE"
  [ "$STATE" = "Completed" ] && break
  [ "$STATE" = "Failed" ]    && { echo "Job failed"; exit 1; }
  sleep 5
done

# ── Step 3: Download results ─────────────────────────────────────────────────
curl -s "$BASE/result/$JOB_ID/" \\
  -H "Authorization: Bearer $API_KEY" \\
  -o output.csv

echo "Results saved to output.csv"`;

const JSON_BODY_EXAMPLE = `import requests

API_KEY = "ak_your_key_here"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type":  "application/json",
}

response = requests.post(
    "${BASE}/submit/",
    headers=HEADERS,
    json={
        "targets":             ["kcat"],
        "methods":             {"kcat": "DLKcat"},
        "handleLongSequences": "truncate",
        "useExperimental":     True,
        "includeSimilarityColumns": True,
        "canonicalizeSubstrates": True,
        "data": [
            {"Protein Sequence": "MKTLLIFAGFCLAGLSLTPVAHA...", "Substrates": "CC(=O)O;O"},
            {"Protein Sequence": "MGSSHHHHHHSSGLVPRGSH...",   "Substrates": "C1CCCCC1;CCO"},
        ],
    },
)
job = response.json()
print(job["jobId"])`;

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function ApiDocs() {
  return (
    <div className="api-docs-page">
      <div className="api-docs">

        {/* ── Hero ────────────────────────────────────────────────────────── */}
        <header className="api-docs-hero">
          <h1>Developer API</h1>
          <div className="api-base-url">
            <span className="api-base-url-label">BASE URL</span>
            <span>{BASE}</span>
          </div>
        </header>

        {/* ── Quick start ─────────────────────────────────────────────────── */}
        <section className="api-docs-section">
          <h2>Quick Start</h2>
          <div className="quick-start-steps">
            <div className="quick-start-step">
              <div className="step-number">1</div>
              <div className="step-text">
                <strong>Generate an API key</strong>
                <span>Use the key generator below to create a Bearer token.</span>
              </div>
            </div>
            <div className="quick-start-step">
              <div className="step-number">2</div>
              <div className="step-text">
                <strong>Submit a job</strong>
                <span>
                  POST your CSV file (or raw data as JSON) to <code>/submit/</code> with
                  the prediction type and method you want.
                </span>
              </div>
            </div>
            <div className="quick-start-step">
              <div className="step-number">3</div>
              <div className="step-text">
                <strong>Poll for status</strong>
                <span>
                  Check <code>/status/&#123;jobId&#125;/</code> until
                  the job is <strong>Completed</strong> (or <strong>Failed</strong>).
                </span>
              </div>
            </div>
            <div className="quick-start-step">
              <div className="step-number">4</div>
              <div className="step-text">
                <strong>Download results</strong>
                <span>
                  Fetch <code>/result/&#123;jobId&#125;/</code> for a CSV file (add{' '}
                  <code>?format=json</code> to receive structured JSON).
                </span>
              </div>
            </div>
          </div>
        </section>

        {/* ── Authentication ──────────────────────────────────────────────── */}
        <section className="api-docs-section">
          <h2>Authentication</h2>
          <p style={{ lineHeight: 1.7, opacity: 0.9 }}>
            All endpoints except <code>/health/</code> and <code>/methods/</code> require
            a Bearer token in the <code>Authorization</code> header:
          </p>
          <div className="auth-example">
            Authorization: Bearer ak_7f8a9b2c3d4e5f6a…
          </div>

          <ApiKeyManager />
        </section>

        {/* ── Endpoints ───────────────────────────────────────────────────── */}
        <section className="api-docs-section">
          <h2>Endpoints</h2>
          <div className="endpoint-list">

            {/* /health/ */}
            <EndpointCard
              method="GET"
              path="/api/v1/health/"
              summary="Service health check"
              requiresAuth={false}
            >
              <p className="endpoint-description">
                Returns a simple confirmation that the service is running. Useful for
                monitoring scripts or pre-flight checks before submitting a batch.
              </p>
              <p className="example-section-label">Request</p>
              <TabbedCode tabs={HEALTH_REQ} />
              <p className="example-section-label">Response</p>
              <CodeBlock language="json" code={`{
  "status": "ok",
  "service": "OpenKineticsPredictor API",
  "version": "1",
  "timestamp": "2026-03-04T14:30:00.000000+00:00"
}`} />
            </EndpointCard>

            {/* /methods/ */}
            <EndpointCard
              method="GET"
              path="/api/v1/methods/"
              summary="List all available prediction methods"
              requiresAuth={false}
            >
              <p className="endpoint-description">
                Returns all available prediction methods, the CSV columns each requires,
                their sequence-length limits, and supported prediction types. No
                authentication needed — use this to discover what to pass to{' '}
                <code>/submit/</code> before writing your script.
              </p>
              <p className="example-section-label">Request</p>
              <TabbedCode tabs={METHODS_REQ} />
              <p className="example-section-label">Response</p>
              <CodeBlock language="json" code={`{
  "methods": {
    "kcat": [
      {
        "id": "DLKcat",
        "supports": ["kcat"],
        "inputFormat": "single",
        "requiredColumns": ["Protein Sequence", "Substrate"],
        "acceptedCsvTypes": ["single", "multi", "full_reaction"],
        "acceptedCsvTypesByTarget": {
          "kcat": ["single", "multi", "full_reaction"]
        },
        "inputBehaviorByTarget": {"kcat": "expanded_pair"},
        "maxSeqLen": null
      },
      {
        "id": "CatPred",
        "supports": ["kcat", "Km"],
        "acceptedCsvTypesByTarget": {
          "kcat": ["multi", "full_reaction"],
          "Km": ["single", "multi", "full_reaction"]
        },
        "inputBehaviorByTarget": {
          "kcat": "native_multi",
          "Km": "expanded_pair"
        }
      }
    ],
    "Km": ["..."],
    "kcat/Km": ["..."]
  },
  "predictionTypes": ["kcat", "Km", "kcat/Km"]
}`} />
            </EndpointCard>

            {/* /quota/ */}
            <EndpointCard
              method="GET"
              path="/api/v1/quota/"
              summary="Check remaining daily quota"
              requiresAuth={true}
            >
              <p className="endpoint-description">
                Returns your current quota usage. The counter resets at midnight UTC.
              </p>
              <p className="example-section-label">Request</p>
              <TabbedCode tabs={QUOTA_REQ} />
              <p className="example-section-label">Response</p>
              <CodeBlock language="json" code={`{
  "limit": 20000,
  "used": 150,
  "remaining": 19850,
  "resetsInSeconds": 36000
}`} />
            </EndpointCard>

            {/* /validate/ */}
            <EndpointCard
              method="POST"
              path="/api/v1/validate/"
              summary="Validate input data without submitting a job"
              requiresAuth={true}
            >
              <p className="endpoint-description">
                Checks substrate SMILES/InChI strings and protein amino-acid sequences
                for validity, and reports which rows would be skipped or truncated by
                each model. No prediction is run and <strong>no quota is consumed</strong>.
              </p>
              <p className="endpoint-description">
                Set <code>runSimilarity=true</code> to also run MMseqs2 sequence
                similarity analysis against each method's training database. This
                is a <strong>synchronous</strong> call that blocks until the analysis
                completes — it can take several minutes for large inputs.
              </p>

              <p className="example-section-label" style={{ marginTop: '1rem' }}>
                Form fields (CSV file upload)
              </p>
              <table className="api-table">
                <thead>
                  <tr>
                    <th>Field</th>
                    <th>Required</th>
                    <th>Values</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td><code>file</code></td>
                    <td>Yes</td>
                    <td>A <code>.csv</code> file</td>
                  </tr>
                  <tr>
                    <td><code>runSimilarity</code></td>
                    <td>No (default: <code>false</code>)</td>
                    <td><code>true</code> · <code>false</code></td>
                  </tr>
                </tbody>
              </table>

              <p className="example-section-label">Request</p>
              <TabbedCode tabs={VALIDATE_REQ} />

              <p className="example-section-label">Response — without similarity</p>
              <CodeBlock language="json" code={`{
  "rowCount": 50,
  "invalidSubstrates": [
    { "row": 2, "value": "NOT_VALID_SMILES", "error": "Could not parse molecule" }
  ],
  "invalidProteins": [
    { "row": 5, "value": "!!!FAKE!!!", "error": "Invalid amino acid characters" }
  ],
  "lengthViolations": {
    "DLKcat": 0,
    "TurNup": 3,
    "EITLEM": 3,
    "UniKP": 1,
    "CataPro": 1,
    "KinForm-H": 0,
    "KinForm-L": 0,
    "CatPred": 0,
    "OmniESI": 0,
    "CatRange": 0,
    "IECata": 1,
    "MMISA-KM": 0,
    "Server": 0
  },
  "lengthLimits": {
    "DLKcat": null,
    "TurNup": 1024,
    "EITLEM": 1024,
    "UniKP": 1000,
    "CataPro": 1000,
    "KinForm-H": 1500,
    "KinForm-L": 1500,
    "CatPred": 2048,
    "OmniESI": 1000,
    "CatRange": 1024,
    "IECata": 1000,
    "MMISA-KM": 500,
    "Server": 10000
  },
  "similarity": null
}`} />
              <div className="api-callout api-callout-info">
                <code>lengthViolations</code> reports counts and <code>lengthLimits</code> reports limits for all
                registered methods (derived from descriptor <code>max_seq_len</code>).
              </div>

              <p className="example-section-label">Response — with <code>runSimilarity=true</code></p>
              <CodeBlock language="json" code={`{
  "rowCount": 50,
  "invalidSubstrates": [],
  "invalidProteins": [],
  "lengthViolations": {},
  "similarity": {
    "DLKcat": {
      "histogram_max":  [0.0, 0.0, 0.0, 2.1, 8.4, 14.6, 45.2, 20.8, 8.9, 0.0],
      "histogram_mean": [0.0, 0.0, 1.5, 5.3, 12.1, 22.4, 38.7, 15.2, 4.8, 0.0],
      "average_max_similarity":  67.3,
      "average_mean_similarity": 54.1,
      "count_max":  [0, 0, 0, 1, 4, 7, 21, 10, 5, 0],
      "count_mean": [0, 0, 1, 3, 6, 11, 19, 7, 3, 0]
    },
    "TurNup": { "...": "same structure" },
    "...": "one entry per method"
  }
}`} />
              <div className="api-callout api-callout-info">
                Each histogram has 10 bins covering 0–9%, 10–19%, …, 90–100%
                sequence identity. Values are percentages of input sequences
                (e.g. <code>45.2</code> means 45.2% of sequences fall in that bin).
              </div>
            </EndpointCard>

            {/* /submit/ */}
            <EndpointCard
              method="POST"
              path="/api/v1/submit/"
              summary="Submit a new prediction job"
              requiresAuth={true}
            >
              <p className="endpoint-description">
                Accepts either a CSV file upload (<code>multipart/form-data</code>) or
                inline JSON data (<code>application/json</code>). Returns a job ID you can
                use to poll status and download results.
              </p>

              <p className="example-section-label" style={{ marginTop: '1rem' }}>
                Form fields (CSV file upload)
              </p>
              <table className="api-table">
                <thead>
                  <tr>
                    <th>Field</th>
                    <th>Required</th>
                    <th>Values</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td><code>file</code></td>
                    <td>Yes</td>
                    <td>A <code>.csv</code> file</td>
                  </tr>
                  <tr>
                    <td><code>targets</code></td>
                    <td>Yes</td>
                    <td>JSON array subset of <code>["kcat","Km","kcat/Km"]</code></td>
                  </tr>
                  <tr>
                    <td><code>methods</code></td>
                    <td>Yes</td>
                    <td>JSON object mapping each selected target to a method key</td>
                  </tr>
                  <tr>
                    <td><code>handleLongSequences</code></td>
                    <td>No (default: <code>truncate</code>)</td>
                    <td><code>truncate</code> · <code>skip</code></td>
                  </tr>
                  <tr>
                    <td><code>useExperimental</code></td>
                    <td>No (default: <code>false</code>)</td>
                    <td><code>true</code> · <code>false</code></td>
                  </tr>
                  <tr>
                    <td><code>canonicalizeSubstrates</code></td>
                    <td>No (default: <code>true</code>)</td>
                    <td><code>true</code> · <code>false</code></td>
                  </tr>
                  <tr>
                    <td><code>includeSimilarityColumns</code></td>
                    <td>No (default: <code>true</code>)</td>
                    <td><code>true</code> · <code>false</code></td>
                  </tr>
                </tbody>
              </table>

              <p className="example-section-label">Request</p>
              <TabbedCode tabs={SUBMIT_REQ} />
              <p className="example-section-label">Response — 201 Created</p>
              <CodeBlock language="json" code={`{
  "jobId": "aB3kX9z",
  "status": "Pending",
  "statusUrl": "/api/v1/status/aB3kX9z/",
  "resultUrl": "/api/v1/result/aB3kX9z/",
  "quota": {
    "limit": 20000,
    "used": 250,
    "remaining": 19750,
    "resetsInSeconds": 36000
  }
}`} />
            </EndpointCard>

            {/* /status/ */}
            <EndpointCard
              method="GET"
              path="/api/v1/status/{jobId}/"
              summary="Poll job status and progress"
              requiresAuth={true}
            >
              <p className="endpoint-description">
                Poll this endpoint until <code>status</code> is <code>"Completed"</code>{' '}
                or <code>"Failed"</code>. A polling interval of 5–10 seconds is reasonable
                for most jobs.
              </p>
              <p className="example-section-label">Request</p>
              <TabbedCode tabs={STATUS_REQ} />
              <p className="example-section-label">Response — Completed</p>
              <CodeBlock language="json" code={`{
  "jobId": "aB3kX9z",
  "status": "Completed",
  "submittedAt": "2026-03-04T14:30:00+00:00",
  "completedAt":  "2026-03-04T14:32:15+00:00",
  "elapsedSeconds": 135,
  "resultUrl": "/api/v1/result/aB3kX9z/",
  "progress": {
    "moleculesTotal": 50,
    "moleculesProcessed": 50,
    "predictionsTotal": 50,
    "predictionsMade": 48,
    "invalidRows": 2
  }
}`} />
              <p className="example-section-label">Response — Failed</p>
              <CodeBlock language="json" code={`{
  "jobId": "aB3kX9z",
  "status": "Failed",
  "submittedAt": "2026-03-04T14:30:00+00:00",
  "elapsedSeconds": 20,
  "error": "DLKcat prediction ran out of memory. Try a smaller batch.",
  "progress": { ... }
}`} />
            </EndpointCard>

            {/* /result/ */}
            <EndpointCard
              method="GET"
              path="/api/v1/result/{jobId}/"
              summary="Download prediction results (CSV or JSON)"
              requiresAuth={true}
            >
              <p className="endpoint-description">
                Returns the output CSV as a file attachment by default. Add{' '}
                <code>?format=json</code> to receive the same data as a JSON object —
                useful for piping results directly into your analysis code without
                writing an intermediate file. Returns <strong>409 Conflict</strong> if
                the job has not yet completed.
              </p>
              <p className="example-section-label">Request</p>
              <TabbedCode tabs={RESULT_REQ} />
              <p className="example-section-label">Response — <code>?format=json</code></p>
              <CodeBlock language="json" code={`{
  "jobId": "aB3kX9z",
  "columns": ["Protein Sequence", "Substrate", "kcat (1/s)", "Source", "Extra Info"],
  "rowCount": 50,
  "data": [
    {
      "Protein Sequence": "MKTL...",
      "Substrate": "CC(=O)O",
      "kcat (1/s)": 12.34,
      "Source": "Prediction from DLKcat",
      "Extra Info": ""
    },
    ...
  ]
}`} />
            </EndpointCard>

          </div>
        </section>

        {/* ── Full code examples ──────────────────────────────────────────── */}
        <section className="api-docs-section">
          <h2>Full Workflow Examples</h2>
          <p style={{ marginBottom: '1rem', opacity: 0.85 }}>
            Complete submit → poll → download workflows you can copy and adapt.
          </p>
          <TabbedCode
            tabs={[
              { label: 'Python', code: PYTHON_FULL },
              { label: 'curl',   code: CURL_FULL },
            ]}
          />
        </section>

        {/* ── JSON body submission ────────────────────────────────────────── */}
        <section className="api-docs-section">
          <h2>JSON Body Submission</h2>
          <p style={{ opacity: 0.85, lineHeight: 1.7 }}>
            For small datasets (&le; 10,000 rows) you can send data directly as JSON
            instead of uploading a file.
          </p>
          <CodeBlock language="python" code={JSON_BODY_EXAMPLE} />
          <div className="api-callout api-callout-info">
            The <code>"data"</code> array must contain objects whose keys are the exact
            column names required by your chosen method (e.g.{' '}
            <code>"Protein Sequence"</code> and either <code>"Substrate"</code> or{' '}
            <code>"Substrates"</code>). Column names
            are case-sensitive.
          </div>
        </section>

        {/* ── CSV format reference ────────────────────────────────────────── */}
        <section className="api-docs-section">
          <h2>CSV Format Reference</h2>
          <p style={{ marginBottom: '1rem', opacity: 0.85 }}>
            Single-substrate methods accept either one <code>Substrate</code> or an ordered,
            semicolon-separated <code>Substrates</code> list. CatPred kcat consumes the complete
            substrate set natively. TurNup requires a full reaction with <code>Products</code>.
            Other methods accept full-reaction files but ignore and preserve the products.
            <code>Protein Sequence</code> may contain semicolon-separated candidate sequences; output
            remains one row per input row.
          </p>
          <div className="api-callout api-callout-info" style={{ marginBottom: '1rem' }}>
            <strong>Handling long protein sequences:</strong> Set <code>handleLongSequences</code> in
            <code>/submit/</code> requests.
            <ul style={{ marginTop: '0.5rem', marginBottom: 0 }}>
              <li>
                <code>truncate</code> (default): keeps over-limit rows and trims each long sequence to
                the method limit by keeping the first and last halves.
              </li>
              <li>
                <code>skip</code>: excludes rows where <code>Protein Sequence</code> exceeds the method limit.
              </li>
            </ul>
          </div>
          <table className="api-table">
            <thead>
              <tr>
                <th>Method</th>
                <th>Predicts</th>
                <th>Required columns</th>
                <th>Max sequence length</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><code>DLKcat</code></td>
                <td>kcat</td>
                <td><code>Protein Sequence</code>, <code>Substrate</code></td>
                <td>No limit</td>
              </tr>
              <tr>
                <td><code>TurNup</code></td>
                <td>kcat</td>
                <td><code>Protein Sequence</code>, <code>Substrates</code>, <code>Products</code></td>
                <td>1,024 residues</td>
              </tr>
              <tr>
                <td><code>EITLEM</code></td>
                <td>kcat or Km</td>
                <td><code>Protein Sequence</code>, <code>Substrate</code></td>
                <td>1,024 residues</td>
              </tr>
              <tr>
                <td><code>UniKP</code></td>
                <td>kcat or Km</td>
                <td><code>Protein Sequence</code>, <code>Substrate</code></td>
                <td>1,000 residues</td>
              </tr>
              <tr>
                <td><code>CataPro</code></td>
                <td>kcat, Km, or kcat/Km</td>
                <td><code>Protein Sequence</code>, <code>Substrate</code></td>
                <td>1,000 residues</td>
              </tr>
              <tr>
                <td><code>KinForm-H</code></td>
                <td>kcat or Km</td>
                <td><code>Protein Sequence</code>, <code>Substrate</code></td>
                <td>1,500 residues</td>
              </tr>
              <tr>
                <td><code>KinForm-L</code></td>
                <td>kcat only</td>
                <td><code>Protein Sequence</code>, <code>Substrate</code></td>
                <td>1,500 residues</td>
              </tr>
              <tr>
                <td><code>CatPred</code></td>
                <td>kcat or Km</td>
                <td>kcat: <code>Protein Sequence</code>, <code>Substrates</code>; Km also accepts <code>Substrate</code></td>
                <td>2,048 residues</td>
              </tr>
              <tr>
                <td><code>OmniESI</code></td>
                <td>kcat or Km</td>
                <td><code>Protein Sequence</code>, <code>Substrate</code></td>
                <td>1,000 residues</td>
              </tr>
              <tr>
                <td><code>CatRange</code></td>
                <td>kcat or Km</td>
                <td><code>Protein Sequence</code>, <code>Substrate</code></td>
                <td>1,024 residues</td>
              </tr>
              <tr>
                <td><code>IECata</code></td>
                <td>kcat/Km</td>
                <td><code>Protein Sequence</code>, <code>Substrate</code></td>
                <td>1,000 residues</td>
              </tr>
              <tr>
                <td><code>MMISA-KM</code></td>
                <td>Km</td>
                <td><code>Protein Sequence</code>, <code>Substrate</code></td>
                <td>500 residues</td>
              </tr>
            </tbody>
          </table>

          <div className="api-callout api-callout-info" style={{ marginTop: '1rem' }}>
            <strong>Substrate format:</strong> Use SMILES strings (e.g.{' '}
            <code>CC(=O)O</code> for acetic acid) or InChI strings. Single inputs use
            one value in <code>Substrate</code>. Ordered lists use semicolons in{' '}
            <code>Substrates</code>, e.g. <code>CC(=O)O;C1CCCCC1</code>. Single-substrate methods
            return the maximum kcat and JSON-encoded Km and kcat/Km arrays; failed positions
            are <code>null</code>. CatPred kcat instead makes one native prediction for the
            complete substrate set. TurNup uses <code>Products</code>; other methods ignore them.
            Supplied products are still validated for every submission.
          </div>
          <div className="api-callout api-callout-info" style={{ marginTop: '1rem' }}>
            <strong>Protein sequence lists:</strong> Separate candidate protein sequences with
            semicolons in <code>Protein Sequence</code>. The service predicts each candidate and
            reduces per input row: max for kcat, min for Km, and max for direct kcat/Km. With
            <code>Substrates</code>, Km and kcat/Km remain JSON arrays ordered by substrate.
          </div>
        </section>

        {/* ── Rate limits ─────────────────────────────────────────────────── */}
        <section className="api-docs-section">
          <h2>Rate Limits</h2>
          <p style={{ lineHeight: 1.7, opacity: 0.9 }}>
            Each user is subject to a daily prediction quota. The default limit is{' '}
            <strong>20,000 input reaction rows per day</strong>, resetting at midnight UTC.
            Expanded per-substrate predictions do not consume extra quota.{' '}
            Custom limits can be arranged for high-throughput workflows.
          </p>
          <p style={{ lineHeight: 1.7, opacity: 0.9 }}>
            Every response from an authenticated endpoint includes the following headers:
          </p>
          <table className="api-table">
            <thead>
              <tr>
                <th>Header</th>
                <th>Description</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><code>X-RateLimit-Limit</code></td>
                <td>Your daily prediction limit</td>
              </tr>
              <tr>
                <td><code>X-RateLimit-Remaining</code></td>
                <td>Predictions remaining today</td>
              </tr>
              <tr>
                <td><code>X-RateLimit-Reset</code></td>
                <td>Seconds until the counter resets</td>
              </tr>
            </tbody>
          </table>
          <p style={{ marginTop: '1rem', opacity: 0.85 }}>
            When your quota is exceeded, the API returns{' '}
            <span className="status-badge status-4xx">429</span> Too Many Requests.
            The unused quota for a failed or partially completed job is automatically
            credited back.
          </p>
        </section>

        {/* ── Error reference ─────────────────────────────────────────────── */}
        <section className="api-docs-section">
          <h2>Error Responses</h2>
          <p style={{ marginBottom: '1rem', opacity: 0.85 }}>
            All errors follow the same format:
          </p>
          <CodeBlock language="json" code={`{ "error": "A human-readable description of what went wrong." }`} />
          <table className="api-table" style={{ marginTop: '1rem' }}>
            <thead>
              <tr>
                <th>Status</th>
                <th>Meaning</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><span className="status-badge status-2xx">201</span></td>
                <td>Job created successfully</td>
              </tr>
              <tr>
                <td><span className="status-badge status-4xx">400</span></td>
                <td>Bad request — invalid parameters, missing columns, or malformed CSV</td>
              </tr>
              <tr>
                <td><span className="status-badge status-4xx">401</span></td>
                <td>Missing or invalid API key</td>
              </tr>
              <tr>
                <td><span className="status-badge status-4xx">403</span></td>
                <td>Account suspended — contact the administrators</td>
              </tr>
              <tr>
                <td><span className="status-badge status-4xx">404</span></td>
                <td>Job ID not found</td>
              </tr>
              <tr>
                <td><span className="status-badge status-4xx">405</span></td>
                <td>Wrong HTTP method</td>
              </tr>
              <tr>
                <td><span className="status-badge status-4xx">409</span></td>
                <td>Results not ready — job has not completed yet</td>
              </tr>
              <tr>
                <td><span className="status-badge status-4xx">429</span></td>
                <td>Daily quota exceeded — try again tomorrow or reduce batch size</td>
              </tr>
              <tr>
                <td><span className="status-badge status-5xx">500</span></td>
                <td>Internal server error — please contact support</td>
              </tr>
            </tbody>
          </table>
        </section>

      </div>
    </div>
  );
}
