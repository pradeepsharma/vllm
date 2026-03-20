---
description: >
  Security best practices for vLLM deployments — API key authentication,
  network security, CORS, endpoint hardening, threat model, and vulnerability
  reporting.
---

# Security best practices

This guide covers the security considerations and hardening steps for
production vLLM deployments. It is organised around the key threat areas:
authentication, transport security, network isolation, endpoint hardening,
container security, and operational security.

!!! tip "Quick reference"
    For a checklist-style summary, see the
    [Production checklist — Security section](production_checklist.md#security).

---

## Threat model

vLLM is designed to be deployed in **trusted network environments**. Several
components — including PyTorch Distributed, KV cache transfer, and inter-node
communication — are **insecure by default** and must be protected by network
isolation.

The primary attack surfaces are:

| Surface | Risk | Mitigation |
|---|---|---|
| HTTP API server | Unauthorised inference, DoS, data exfiltration | API key auth, reverse proxy, firewall |
| Inter-node communication | Unauthenticated access, RCE via PyTorch Distributed | Network isolation, firewall |
| KV cache transfer | Unauthenticated data access | Private network, firewall |
| Container image | Supply chain attacks | Image pinning, vulnerability scanning |
| Model weights | Malicious model execution | Trusted sources, integrity verification |
| Tool server / code execution | SSRF, arbitrary code execution | Strict isolation, allowlists |

---

## API key authentication

### Enabling API key authentication

vLLM's OpenAI-compatible server supports bearer token authentication via the
`--api-key` flag. When set, all requests to `/v1/*` endpoints must include a
valid key in the `Authorization` header.

```bash
vllm serve mistralai/Mistral-7B-Instruct-v0.3 \
    --api-key sk-my-secret-key \
    --host 0.0.0.0 \
    --port 8000
```

Clients must include the key as a bearer token:

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Authorization: Bearer sk-my-secret-key" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "mistralai/Mistral-7B-Instruct-v0.3",
        "messages": [{"role": "user", "content": "Hello"}]
    }'
```

### Multiple API keys

Pass `--api-key` multiple times to accept any of several keys. This is useful
for key rotation without downtime:

```bash
vllm serve mistralai/Mistral-7B-Instruct-v0.3 \
    --api-key sk-key-v1 \
    --api-key sk-key-v2 \
    --host 0.0.0.0 \
    --port 8000
```

### Using environment variables

Avoid passing secrets on the command line (they appear in process listings).
Use the `VLLM_API_KEY` environment variable instead:

```bash
export VLLM_API_KEY="sk-my-secret-key"
vllm serve mistralai/Mistral-7B-Instruct-v0.3 \
    --host 0.0.0.0 \
    --port 8000
```

In Kubernetes, inject the key from a Secret:

```yaml
env:
  - name: VLLM_API_KEY
    valueFrom:
      secretKeyRef:
        name: vllm-api-keys
        key: api-key
```

Create the Secret:

```bash
kubectl create secret generic vllm-api-keys \
    --from-literal=api-key=sk-my-secret-key \
    --namespace ns-vllm
```

### API key authentication limitations

!!! warning "Critical: API key does not protect all endpoints"
    The `--api-key` flag only protects endpoints under the `/v1` path prefix.
    Many other endpoints on the same HTTP server are **unauthenticated by
    default**. Do not rely exclusively on `--api-key` for production security.

#### Protected endpoints (require API key)

When `--api-key` is configured, the following endpoints require a valid bearer
token:

| Endpoint | Description |
|---|---|
| `GET /v1/models` | List available models |
| `POST /v1/chat/completions` | Chat completions |
| `POST /v1/completions` | Text completions |
| `POST /v1/embeddings` | Generate embeddings |
| `POST /v1/audio/transcriptions` | Audio transcription |
| `POST /v1/audio/translations` | Audio translation |
| `POST /v1/messages` | Anthropic-compatible messages API |
| `POST /v1/responses` | Response management |
| `POST /v1/score` | Scoring API |
| `POST /v1/rerank` | Reranking API |

#### Unprotected endpoints (no API key required)

The following endpoints are **always accessible** without authentication, even
when `--api-key` is configured:

**Inference endpoints (bypass authentication):**

| Endpoint | Risk |
|---|---|
| `POST /invocations` | SageMaker-compatible endpoint — same inference as `/v1` |
| `POST /inference/v1/generate` | Generate completions |
| `POST /pooling` | Pooling API |
| `POST /classify` | Classification API |
| `POST /score` | Scoring API (non-`/v1` variant) |
| `POST /rerank` | Reranking API (non-`/v1` variant) |

**Operational control endpoints (always enabled):**

| Endpoint | Risk |
|---|---|
| `POST /pause` | Pause generation — causes denial of service |
| `POST /resume` | Resume generation |
| `POST /scale_elastic_ep` | Trigger scaling operations |

**Utility endpoints:**

| Endpoint | Notes |
|---|---|
| `POST /tokenize` | Tokenize text |
| `POST /detokenize` | Detokenize tokens |
| `GET /health` | Health check |
| `GET /ping` | SageMaker health check |
| `GET /version` | Version information |
| `GET /load` | Server load metrics |

**Conditionally enabled endpoints:**

| Endpoint | Condition | Risk |
|---|---|---|
| `GET /tokenizer_info` | `--enable-tokenizer-info-endpoint` flag | Exposes chat templates and tokenizer config |
| `POST /server_info` | `VLLM_SERVER_DEV_MODE=1` | Exposes detailed server configuration |
| `POST /reset_prefix_cache` | `VLLM_SERVER_DEV_MODE=1` | Disrupts service |
| `POST /reset_mm_cache` | `VLLM_SERVER_DEV_MODE=1` | Disrupts service |
| `POST /reset_encoder_cache` | `VLLM_SERVER_DEV_MODE=1` | Disrupts service |
| `POST /sleep` | `VLLM_SERVER_DEV_MODE=1` | Causes denial of service |
| `POST /wake_up` | `VLLM_SERVER_DEV_MODE=1` | — |
| `POST /collective_rpc` | `VLLM_SERVER_DEV_MODE=1` | **Extremely dangerous** — arbitrary RPC execution |
| `POST /start_profile` | Profiling enabled | Development only |
| `POST /stop_profile` | Profiling enabled | Development only |

!!! danger "Never enable `VLLM_SERVER_DEV_MODE=1` in production"
    Development mode exposes `/collective_rpc`, which allows arbitrary RPC
    execution on the engine. This is an extremely high-severity vulnerability
    if exposed to untrusted networks.

### Recommended approach: reverse proxy allowlist

The most effective way to secure vLLM's HTTP surface is to deploy it behind a
reverse proxy that **explicitly allowlists** only the endpoints you want to
expose:

```nginx
# nginx.conf — only expose /v1/* endpoints
server {
    listen 443 ssl;
    server_name api.example.com;

    # Only allow /v1/* paths
    location /v1/ {
        proxy_pass http://vllm-backend:8000;
        proxy_set_header Authorization $http_authorization;
        proxy_set_header Host $host;
    }

    # Block everything else
    location / {
        return 403;
    }
}
```

---

## Transport security (TLS/HTTPS)

Always encrypt traffic between clients and the vLLM server. See the dedicated
[SSL/TLS configuration guide](ssl_tls.md) for full details. Key points:

- **Never serve plain HTTP on a public network.** Use `--ssl-keyfile` and
  `--ssl-certfile` to enable HTTPS directly, or terminate TLS at a reverse
  proxy or load balancer.
- **Use certificates from a trusted CA.** Self-signed certificates are
  acceptable only for development.
- **Enable automatic certificate renewal.** Use `--enable-ssl-refresh` so
  vLLM reloads certificates without a restart when they are renewed by
  Certbot or cert-manager.
- **Enforce TLS 1.2 or higher.** Disable SSLv3, TLS 1.0, and TLS 1.1 at the
  load balancer or Nginx configuration.
- **Prefer ECDHE cipher suites** for forward secrecy.

Quick start:

```bash
vllm serve mistralai/Mistral-7B-Instruct-v0.3 \
    --ssl-keyfile /etc/ssl/private/vllm.key \
    --ssl-certfile /etc/ssl/certs/vllm.crt \
    --enable-ssl-refresh \
    --host 0.0.0.0 \
    --port 8443
```

---

## Network security

### Firewall configuration

vLLM's internal communication channels (PyTorch Distributed, KV cache
transfer) are **insecure by default** — they accept connections from any host
without authentication or encryption. Protect them with a firewall.

**Minimum firewall rules for a single-node deployment:**

```bash
# Allow only the API port from external networks
iptables -A INPUT -p tcp --dport 8000 -j ACCEPT

# Block all other incoming connections
iptables -A INPUT -j DROP
```

**For multi-node deployments**, restrict inter-node ports to the cluster's
private network:

```bash
# Allow PyTorch Distributed (default port range)
iptables -A INPUT -s 10.0.0.0/8 -p tcp --dport 29500 -j ACCEPT

# Allow KV cache transfer
iptables -A INPUT -s 10.0.0.0/8 -p tcp --dport 14579 -j ACCEPT

# Block these ports from external networks
iptables -A INPUT -p tcp --dport 29500 -j DROP
iptables -A INPUT -p tcp --dport 14579 -j DROP
```

### Network isolation for multi-node deployments

All inter-node communication in vLLM is **unencrypted and unauthenticated**:

| Communication type | Default port | Security |
|---|---|---|
| PyTorch Distributed (TCP) | 29500 | None — no auth, no encryption |
| KV cache transfer | 14579 | None — no auth, no encryption |
| Data parallel master | 29500 | None — no auth, no encryption |

**Recommendations:**

1. **Deploy on an isolated private network.** Use a VPC, VLAN, or dedicated
   network segment that is not reachable from the public internet.
2. **Use network segmentation.** Separate the inference cluster from other
   workloads.
3. **Configure `VLLM_HOST_IP`** to bind inter-node communication to a specific
   private interface:

    ```bash
    export VLLM_HOST_IP=10.0.1.5
    ```

4. **Configure specific KV cache transfer addresses:**

    ```bash
    vllm serve ... \
        --kv-ip 10.0.1.5 \
        --kv-port 14579
    ```

For detailed multi-node security guidance, see the
[PyTorch Security Policy](https://github.com/pytorch/pytorch/blob/main/SECURITY.md).

### Binding to specific interfaces

By default, vLLM binds to all interfaces (`0.0.0.0`). In production, bind to
a specific interface or use a Unix domain socket:

```bash
# Bind to a specific private IP
vllm serve ... --host 10.0.1.5 --port 8000

# Use a Unix domain socket (for local reverse proxy)
vllm serve ... --uds /run/vllm/vllm.sock
```

When using a Unix domain socket, the reverse proxy (Nginx, Envoy) communicates
with vLLM over the socket, and only the proxy is exposed to the network.

---

## CORS configuration

Cross-Origin Resource Sharing (CORS) controls which web origins can make
requests to the vLLM API. The defaults are permissive — restrict them in
production.

### Default values (permissive — change in production)

| Argument | Default | Production recommendation |
|---|---|---|
| `--allowed-origins` | `["*"]` | `["https://app.example.com"]` |
| `--allowed-methods` | `["*"]` | `["GET", "POST"]` |
| `--allowed-headers` | `["*"]` | `["Authorization", "Content-Type"]` |
| `--allow-credentials` | `false` | Set to `true` only if needed |

### Restricting CORS

```bash
vllm serve mistralai/Mistral-7B-Instruct-v0.3 \
    --allowed-origins '["https://app.example.com", "https://admin.example.com"]' \
    --allowed-methods '["GET", "POST"]' \
    --allowed-headers '["Authorization", "Content-Type"]' \
    --host 0.0.0.0 \
    --port 8000
```

!!! note
    CORS restrictions are enforced by the browser. They do not prevent
    server-to-server requests (e.g., from `curl` or backend services). Use
    API key authentication and network-level controls for server-to-server
    security.

---

## Endpoint hardening

### Disable the FastAPI documentation UI

The Swagger UI and ReDoc endpoints expose your full API schema. Disable them
in production:

```bash
vllm serve ... --disable-fastapi-docs
```

### Do not enable the tokenizer info endpoint

The `/tokenizer_info` endpoint exposes chat templates and tokenizer
configuration, which may reveal prompt engineering strategies or other
implementation details. Only enable it if explicitly required:

```bash
# Only enable if you need to expose tokenizer info
vllm serve ... --enable-tokenizer-info-endpoint
```

### Restrict media URL domains (multimodal deployments)

For multimodal deployments that accept image or audio URLs, restrict the
domains vLLM can fetch from to prevent Server-Side Request Forgery (SSRF)
attacks:

```bash
vllm serve ... \
    --allowed-media-domains upload.wikimedia.org github.com cdn.example.com
```

Also disable HTTP redirect following to prevent bypass:

```bash
export VLLM_MEDIA_URL_ALLOW_REDIRECTS=0
```

### Limit HTTP header size

vLLM includes configurable limits on HTTP header size and count to mitigate
header abuse attacks:

```bash
vllm serve ... \
    --h11-max-incomplete-event-size 1048576 \
    --h11-max-header-count 100
```

| Argument | Default | Description |
|---|---|---|
| `--h11-max-incomplete-event-size` | `4194304` (4 MB) | Maximum size of an incomplete HTTP event |
| `--h11-max-header-count` | `256` | Maximum number of HTTP headers per request |

---

## Container security

### Pin image versions

Always use a specific version tag rather than `latest` to ensure
reproducibility and avoid unexpected changes:

```bash
# Good
docker pull vllm/vllm-openai:v0.9.0

# Avoid in production
docker pull vllm/vllm-openai:latest
```

### Scan images for vulnerabilities

Integrate a container vulnerability scanner into your CI/CD pipeline:

```bash
# Trivy
trivy image vllm/vllm-openai:v0.9.0

# Grype
grype vllm/vllm-openai:v0.9.0
```

### Kubernetes security context

Apply a restrictive security context to the vLLM pod:

```yaml
securityContext:
  runAsNonRoot: false          # vLLM requires root for GPU access
  readOnlyRootFilesystem: true # Mount writable volumes explicitly
  allowPrivilegeEscalation: false
  capabilities:
    drop:
      - ALL
    add:
      - SYS_PTRACE            # Required for GPU profiling (optional)
```

Mount writable directories as volumes:

```yaml
volumeMounts:
  - name: tmp
    mountPath: /tmp
  - name: model-cache
    mountPath: /root/.cache/huggingface

volumes:
  - name: tmp
    emptyDir: {}
  - name: model-cache
    persistentVolumeClaim:
      claimName: model-cache-pvc
```

### Never embed secrets in images

Do not bake API keys, certificates, or Hugging Face tokens into container
images. Use Kubernetes Secrets or a secrets manager:

```yaml
env:
  - name: VLLM_API_KEY
    valueFrom:
      secretKeyRef:
        name: vllm-secrets
        key: api-key
  - name: HF_TOKEN
    valueFrom:
      secretKeyRef:
        name: vllm-secrets
        key: hf-token
```

---

## Tool server security

vLLM supports connecting to external tool servers via `--tool-server`. This
enables models to call tools through the Responses API (`/v1/responses`).

!!! warning "No tool servers are enabled by default"
    Tool servers must be explicitly opted into via `--tool-server`. Do not
    enable them unless your deployment requires tool calling.

### Built-in demo tools (GPT-OSS)

Passing `--tool-server demo` enables built-in demo tools:

- **Code interpreter** (`python`): Executes model-generated Python code in a
  Docker container.
- **Web browser** (`browser`): Searches the web via the Exa API.

#### Code interpreter security risks

The code interpreter Docker container is **not network-isolated by default**.
It inherits the host's Docker networking, which means:

- The container can access the host network and LAN.
- Cloud metadata services (e.g., `169.254.169.254`) may be accessible.
- Internal services reachable from the container may be exploited via SSRF.
- Adversarial inputs (prompt injection) can influence the code being executed.

**Mitigations:**

1. Run the Docker container with `--network none` or a restricted network.
2. Block access to cloud metadata endpoints at the firewall level.
3. Use a custom code execution sandbox with stricter isolation for production.
4. Control which built-in tools are available via
   `VLLM_GPT_OSS_SYSTEM_TOOL_MCP_LABELS`:

    ```bash
    # Only allow web search — disable code interpreter
    export VLLM_GPT_OSS_SYSTEM_TOOL_MCP_LABELS=web_search_preview
    ```

    Valid values: `container`, `code_interpreter`, `web_search_preview`.

---

## Secrets management

### Store secrets in a secrets manager

Never store API keys, certificates, or model access tokens in:

- Container images
- Kubernetes ConfigMaps
- Environment files committed to version control
- Command-line arguments (visible in process listings)

Use one of:

- **Kubernetes Secrets** (base64-encoded, RBAC-protected)
- **AWS Secrets Manager** with IRSA (IAM Roles for Service Accounts)
- **HashiCorp Vault** with Kubernetes auth
- **GCP Secret Manager** with Workload Identity
- **Azure Key Vault** with Managed Identity

### Key rotation

Rotate API keys regularly without downtime by using multiple keys:

```bash
# During rotation: accept both old and new keys
vllm serve ... \
    --api-key sk-old-key \
    --api-key sk-new-key

# After all clients have migrated to the new key:
vllm serve ... \
    --api-key sk-new-key
```

---

## Logging and audit

### Access logging

vLLM uses Uvicorn's access logging by default. In production, configure
logging appropriately:

```bash
# Reduce log verbosity in production
vllm serve ... --uvicorn-log-level warning

# Suppress high-frequency health check logs
vllm serve ... \
    --disable-access-log-for-endpoints "/health,/metrics,/ping"
```

### Request logging

Enable request logging for audit purposes:

```bash
vllm serve ... --enable-log-requests
```

!!! caution
    Request logs may contain sensitive user data (prompts and completions).
    Ensure logs are stored securely and access is restricted.

### Request ID headers

Enable request ID headers for end-to-end tracing and audit correlation:

```bash
vllm serve ... --enable-request-id-headers
```

This adds an `X-Request-Id` header to every response, which can be correlated
with access logs.

### Do not enable stack traces in production

The `--log-error-stack` flag (or `VLLM_SERVER_DEV_MODE=1`) logs full Python
stack traces in error responses. This can expose internal implementation
details to clients. Ensure it is disabled in production:

```bash
# Ensure VLLM_SERVER_DEV_MODE is not set
unset VLLM_SERVER_DEV_MODE
```

---

## Vulnerability reporting

vLLM follows a responsible disclosure process for security vulnerabilities.

### Reporting a vulnerability

Report security issues **privately** using the
[GitHub vulnerability submission form](https://github.com/vllm-project/vllm/security/advisories/new).
Do not open a public GitHub issue for security vulnerabilities.

### Severity categories

| Severity | CVSS Score | Examples |
|---|---|---|
| **Critical** | ≥ 9.0 | Remote code execution without authentication, full system compromise |
| **High** | 7.0 – 8.9 | RCE in specific contexts, significant data loss with some trust required |
| **Moderate** | 4.0 – 6.9 | Denial of service, partial disruption without code execution |
| **Low** | < 4.0 | Informational disclosures, non-exploitable flaws, side-channel attacks |

### Issue triage

Reports are triaged by the
[vulnerability management team](https://docs.vllm.ai/en/latest/contributing/vulnerability_management.html).

### Prenotification policy

For Critical, High, or Moderate severity issues, vLLM may prenotify
organisations that ship or deploy vLLM before public disclosure. To join the
prenotification group, contact the vulnerability management team. Eligibility
requires at least one of:

- Substantial internal deployment of upstream vLLM
- Established internal security teams and compliance measures
- Active and consistent contributions to the upstream vLLM project

---

## Security hardening checklist

Use this checklist to verify your deployment is hardened:

### Authentication

- [ ] `--api-key` is set (or `VLLM_API_KEY` environment variable)
- [ ] API keys are stored in a secrets manager, not in environment files or
  command-line arguments
- [ ] Multiple API keys are configured to allow zero-downtime rotation
- [ ] A reverse proxy allowlists only the endpoints you want to expose

### Transport security

- [ ] TLS is enabled (`--ssl-keyfile`, `--ssl-certfile`) or terminated at a
  reverse proxy
- [ ] Certificates are from a trusted CA (not self-signed)
- [ ] `--enable-ssl-refresh` is set for automatic certificate reload
- [ ] TLS 1.2 or higher is enforced

### Network security

- [ ] vLLM is bound to a specific interface or Unix domain socket
- [ ] Firewall blocks all ports except the API port from external networks
- [ ] Inter-node ports (PyTorch Distributed, KV cache) are restricted to the
  private cluster network
- [ ] `VLLM_HOST_IP` is set to a specific private IP

### Endpoint hardening

- [ ] `--disable-fastapi-docs` is set
- [ ] `VLLM_SERVER_DEV_MODE` is **not** set
- [ ] `--enable-tokenizer-info-endpoint` is **not** set (unless required)
- [ ] `--allowed-media-domains` is configured for multimodal deployments
- [ ] CORS origins are restricted to your application's domain

### Container security

- [ ] Image version is pinned (not `latest`)
- [ ] Image is scanned for vulnerabilities
- [ ] Secrets are injected via Kubernetes Secrets or a secrets manager
- [ ] No secrets are embedded in the container image or ConfigMaps

### Operational security

- [ ] `VLLM_SERVER_DEV_MODE` is not set in production
- [ ] Profiling endpoints are not enabled in production
- [ ] Request logs are stored securely with restricted access
- [ ] Log verbosity is appropriate for production (`--uvicorn-log-level warning`)

---

## Next steps

- [SSL/TLS configuration](ssl_tls.md) — enable HTTPS and configure certificates
- [Production checklist](production_checklist.md) — full production readiness
  checklist
- [Load balancing](load_balancing.md) — reverse proxy and load balancer setup
- [Kubernetes deployment](kubernetes.md) — Kubernetes Secrets and RBAC
- [Monitoring](monitoring.md) — audit logging and observability
