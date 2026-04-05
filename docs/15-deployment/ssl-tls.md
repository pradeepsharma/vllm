# SSL/TLS Configuration

vLLM supports HTTPS for the OpenAI-compatible API server via Uvicorn's SSL support. It also provides automatic certificate hot-reloading through the `SSLCertRefresher` class, which watches certificate files for changes and reloads them without restarting the server.

Source: `vllm/entrypoints/ssl.py`, `vllm/entrypoints/launcher.py`, `vllm/entrypoints/openai/cli_args.py`

## CLI Arguments

SSL is configured via `vllm serve` CLI arguments (defined in `vllm/entrypoints/openai/cli_args.py`):

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--ssl-keyfile` | `str` | `None` | Path to the SSL private key file |
| `--ssl-certfile` | `str` | `None` | Path to the SSL certificate file |
| `--ssl-ca-certs` | `str` | `None` | Path to the CA certificates file (for client cert verification) |
| `--ssl-cert-reqs` | `int` | `0` (CERT_NONE) | Client certificate requirement (stdlib `ssl` module values) |
| `--ssl-ciphers` | `str` | `None` | SSL cipher suites (TLS 1.2 and below only) |
| `--enable-ssl-refresh` | flag | `False` | Enable automatic certificate hot-reload on file change |

## Basic HTTPS Setup

### Step 1: Generate a Self-Signed Certificate (Development)

```bash
# Generate private key and self-signed certificate
openssl req -x509 -newkey rsa:4096 \
    -keyout server.key \
    -out server.crt \
    -days 365 \
    -nodes \
    -subj "/CN=localhost"
```

### Step 2: Start vLLM with SSL

```bash
vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct \
    --ssl-keyfile server.key \
    --ssl-certfile server.crt \
    --host 0.0.0.0 \
    --port 8443
```

### Step 3: Test the HTTPS Endpoint

```bash
# With self-signed cert (skip verification)
curl -k https://localhost:8443/v1/models

# With CA cert
curl --cacert ca.crt https://localhost:8443/v1/models
```

## Mutual TLS (mTLS)

For client certificate authentication, use `--ssl-ca-certs` and `--ssl-cert-reqs`:

```bash
# Generate CA key and certificate
openssl genrsa -out ca.key 4096
openssl req -x509 -new -nodes -key ca.key \
    -sha256 -days 1024 \
    -out ca.crt \
    -subj "/CN=My CA"

# Generate server key and CSR
openssl genrsa -out server.key 4096
openssl req -new -key server.key \
    -out server.csr \
    -subj "/CN=vllm-server"

# Sign server certificate with CA
openssl x509 -req -in server.csr \
    -CA ca.crt -CAkey ca.key \
    -CAcreateserial \
    -out server.crt \
    -days 365 -sha256

# Generate client key and certificate
openssl genrsa -out client.key 4096
openssl req -new -key client.key \
    -out client.csr \
    -subj "/CN=vllm-client"
openssl x509 -req -in client.csr \
    -CA ca.crt -CAkey ca.key \
    -CAcreateserial \
    -out client.crt \
    -days 365 -sha256
```

Start vLLM with mTLS:

```bash
vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct \
    --ssl-keyfile server.key \
    --ssl-certfile server.crt \
    --ssl-ca-certs ca.crt \
    --ssl-cert-reqs 2 \
    --host 0.0.0.0 \
    --port 8443
```

The `--ssl-cert-reqs` values correspond to Python's `ssl` module:

| Value | Constant | Description |
|-------|----------|-------------|
| `0` | `ssl.CERT_NONE` | No client certificate required (default) |
| `1` | `ssl.CERT_OPTIONAL` | Client certificate requested but not required |
| `2` | `ssl.CERT_REQUIRED` | Client certificate required and verified |

Test with client certificate:

```bash
curl --cacert ca.crt \
     --cert client.crt \
     --key client.key \
     https://localhost:8443/v1/models
```

## Certificate Hot-Reload (`SSLCertRefresher`)

The `SSLCertRefresher` class (`vllm/entrypoints/ssl.py`) monitors certificate files using `watchfiles` and reloads them when they change — without restarting the server. This is essential for production deployments using short-lived certificates (e.g., Let's Encrypt, cert-manager).

### Enabling Hot-Reload

```bash
vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct \
    --ssl-keyfile /etc/ssl/server.key \
    --ssl-certfile /etc/ssl/server.crt \
    --ssl-ca-certs /etc/ssl/ca.crt \
    --enable-ssl-refresh \
    --host 0.0.0.0 \
    --port 8443
```

### How It Works

The `SSLCertRefresher` is initialized in `vllm/entrypoints/launcher.py` when `enable_ssl_refresh=True`:

```python
ssl_cert_refresher = (
    None
    if not enable_ssl_refresh
    else SSLCertRefresher(
        ssl_context=config.ssl,
        key_path=config.ssl_keyfile,
        cert_path=config.ssl_certfile,
        ca_path=config.ssl_ca_certs,
    )
)
```

The `SSLCertRefresher` creates two async file watchers:

1. **Certificate chain watcher** — monitors `ssl_keyfile` and `ssl_certfile`:
   ```python
   def update_ssl_cert_chain(change: Change, file_path: str) -> None:
       logger.info("Reloading SSL certificate chain")
       self.ssl.load_cert_chain(self.cert_path, self.key_path)
   ```

2. **CA certificates watcher** — monitors `ssl_ca_certs`:
   ```python
   def update_ssl_ca(change: Change, file_path: str) -> None:
       logger.info("Reloading SSL CA certificates")
       self.ssl.load_verify_locations(self.ca_path)
   ```

When a file change is detected, the SSL context is updated in-place. Active connections continue using the old certificate; new connections use the updated certificate.

### `SSLCertRefresher` API

```python
class SSLCertRefresher:
    def __init__(
        self,
        ssl_context: SSLContext,
        key_path: str | None = None,
        cert_path: str | None = None,
        ca_path: str | None = None,
    ) -> None: ...

    def stop(self) -> None:
        """Stop watching files and cancel background tasks."""
```

## SSL in Docker

Mount certificate files into the container:

```bash
docker run --runtime nvidia --gpus all \
    -v /etc/ssl/vllm:/etc/ssl/vllm:ro \
    -p 8443:8443 \
    vllm/vllm-openai:latest \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --ssl-keyfile /etc/ssl/vllm/server.key \
    --ssl-certfile /etc/ssl/vllm/server.crt \
    --enable-ssl-refresh \
    --host 0.0.0.0 \
    --port 8443
```

## SSL in Kubernetes

Use cert-manager to automatically provision and renew certificates:

```yaml
apiVersion: cert-manager.io/v1
kind: Certificate
metadata:
  name: vllm-tls
spec:
  secretName: vllm-tls-secret
  issuerRef:
    name: letsencrypt-prod
    kind: ClusterIssuer
  dnsNames:
    - vllm.example.com
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-server
spec:
  template:
    spec:
      containers:
        - name: vllm
          args:
            - --ssl-keyfile
            - /etc/ssl/tls.key
            - --ssl-certfile
            - /etc/ssl/tls.crt
            - --enable-ssl-refresh   # Hot-reload when cert-manager renews
          volumeMounts:
            - name: tls
              mountPath: /etc/ssl
              readOnly: true
      volumes:
        - name: tls
          secret:
            secretName: vllm-tls-secret
```

With `--enable-ssl-refresh`, vLLM automatically picks up renewed certificates when cert-manager updates the Kubernetes Secret and the mounted files change.

## Cipher Suite Configuration

Restrict cipher suites for TLS 1.2 and below:

```bash
vllm serve mymodel \
    --ssl-keyfile server.key \
    --ssl-certfile server.crt \
    --ssl-ciphers "ECDHE-RSA-AES256-GCM-SHA384:ECDHE-RSA-CHACHA20-POLY1305" \
    --host 0.0.0.0 \
    --port 8443
```

> **Note:** `--ssl-ciphers` only affects TLS 1.2 and below. TLS 1.3 cipher suites are not configurable via this option.

## API Key Authentication

Combine SSL with API key authentication for secure access:

```bash
vllm serve mymodel \
    --ssl-keyfile server.key \
    --ssl-certfile server.crt \
    --api-key my-secret-key \
    --host 0.0.0.0 \
    --port 8443
```

Clients must include the key in the `Authorization` header:

```bash
curl https://localhost:8443/v1/models \
    -H "Authorization: Bearer my-secret-key" \
    --cacert ca.crt
```

See [Auth & SSL Reference](../12-api-reference/auth-ssl.md) for full authentication documentation.

## Related Pages

- [Auth & SSL Reference](../12-api-reference/auth-ssl.md) — API key authentication
- [Docker Deployment](docker.md) — mounting certificates in Docker
- [Kubernetes Deployment](kubernetes.md) — cert-manager integration
- [API Reference](../12-api-reference/README.md) — endpoint documentation
