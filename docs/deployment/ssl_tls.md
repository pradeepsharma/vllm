---
description: >
  Configure SSL/TLS for the vLLM API server — certificate setup, mutual TLS,
  cipher suites, and automatic certificate hot-reload.
---

# SSL/TLS configuration

vLLM's OpenAI-compatible server supports HTTPS via Uvicorn's native TLS
integration. You can configure server certificates, CA verification for mutual
TLS (mTLS), cipher suite restrictions, and automatic certificate hot-reload
without restarting the server.

---

## Overview

TLS termination can happen at two layers:

1. **At vLLM directly** — vLLM's Uvicorn server loads your certificate and
   private key and terminates TLS itself. This is the simplest approach for
   single-instance deployments.
1. **At a reverse proxy or load balancer** — Nginx, HAProxy, or a cloud load
   balancer terminates TLS and forwards plain HTTP to vLLM. This is the
   recommended approach for production deployments with multiple instances.

This page covers option 1. For option 2, see [Load balancing](load_balancing.md).

---

## CLI arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--ssl-keyfile` | `str` | `None` | Path to the PEM-encoded private key file |
| `--ssl-certfile` | `str` | `None` | Path to the PEM-encoded certificate file |
| `--ssl-ca-certs` | `str` | `None` | Path to the CA certificate bundle (for mTLS) |
| `--ssl-cert-reqs` | `int` | `0` (CERT_NONE) | Client certificate requirement level |
| `--ssl-ciphers` | `str` | `None` | Colon-separated cipher suite list (TLS 1.2 and below) |
| `--enable-ssl-refresh` | flag | `False` | Watch certificate files and reload on change |

---

## Basic HTTPS setup

### Step 1: Generate a self-signed certificate (development only)

```bash
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem \
    -days 365 -nodes \
    -subj "/CN=localhost"
```

!!! warning
    Self-signed certificates are suitable for development and testing only.
    Use certificates from a trusted CA (e.g., Let's Encrypt, your organisation's
    PKI) in production.

### Step 2: Start vLLM with TLS

```bash
vllm serve mistralai/Mistral-7B-Instruct-v0.3 \
    --ssl-keyfile key.pem \
    --ssl-certfile cert.pem \
    --host 0.0.0.0 \
    --port 8443
```

The server logs will confirm HTTPS is active:

```text
INFO:     Started server process [12345]
INFO:     Uvicorn running on https://0.0.0.0:8443 (Press CTRL+C to quit)
```

### Step 3: Test the connection

```bash
# With a trusted certificate
curl https://your-server:8443/health

# With a self-signed certificate (skip verification — development only)
curl -k https://localhost:8443/health
```

---

## Production certificate setup

### Using Let's Encrypt with Certbot

```bash
# Install Certbot
sudo apt-get install certbot

# Obtain a certificate (standalone mode — stops any service on port 80)
sudo certbot certonly --standalone -d api.example.com

# Certificates are stored at:
# /etc/letsencrypt/live/api.example.com/fullchain.pem
# /etc/letsencrypt/live/api.example.com/privkey.pem
```

Start vLLM with the Let's Encrypt certificate:

```bash
vllm serve mistralai/Mistral-7B-Instruct-v0.3 \
    --ssl-keyfile /etc/letsencrypt/live/api.example.com/privkey.pem \
    --ssl-certfile /etc/letsencrypt/live/api.example.com/fullchain.pem \
    --host 0.0.0.0 \
    --port 443
```

### Using an organisational CA

```bash
vllm serve mistralai/Mistral-7B-Instruct-v0.3 \
    --ssl-keyfile /etc/ssl/private/vllm.key \
    --ssl-certfile /etc/ssl/certs/vllm.crt \
    --host 0.0.0.0 \
    --port 8443
```

---

## Automatic certificate hot-reload

vLLM can watch certificate files for changes and reload them without
restarting the server. This is implemented by `SSLCertRefresher` in
`vllm/entrypoints/ssl.py`.

Enable hot-reload with the `--enable-ssl-refresh` flag:

```bash
vllm serve mistralai/Mistral-7B-Instruct-v0.3 \
    --ssl-keyfile /etc/ssl/private/vllm.key \
    --ssl-certfile /etc/ssl/certs/vllm.crt \
    --enable-ssl-refresh \
    --host 0.0.0.0 \
    --port 8443
```

### How it works

`SSLCertRefresher` uses `watchfiles` to monitor the certificate and key files
asynchronously. When a change is detected:

- **Certificate/key changes** — calls `ssl_context.load_cert_chain()` to
  reload the server certificate and private key.
- **CA certificate changes** — calls `ssl_context.load_verify_locations()` to
  reload the CA bundle used for client verification.

The watcher runs as an asyncio background task and does not block request
processing.

### Typical use case: Let's Encrypt auto-renewal

Let's Encrypt certificates expire every 90 days. With `--enable-ssl-refresh`,
Certbot's renewal hook can simply replace the certificate files and vLLM will
pick up the new certificate automatically:

```bash
# /etc/letsencrypt/renewal-hooks/deploy/reload-vllm.sh
#!/bin/bash
# vLLM detects the file change automatically — no action needed.
# This hook is a no-op when --enable-ssl-refresh is active.
echo "Certificate renewed. vLLM will reload automatically."
```

---

## Mutual TLS (mTLS)

Mutual TLS requires clients to present a valid certificate signed by a trusted
CA. This is useful for service-to-service authentication.

### Server configuration

```bash
vllm serve mistralai/Mistral-7B-Instruct-v0.3 \
    --ssl-keyfile /etc/ssl/private/vllm.key \
    --ssl-certfile /etc/ssl/certs/vllm.crt \
    --ssl-ca-certs /etc/ssl/certs/client-ca.crt \
    --ssl-cert-reqs 2 \
    --host 0.0.0.0 \
    --port 8443
```

### `--ssl-cert-reqs` values

These correspond to Python's `ssl` module constants:

| Value | Constant | Behaviour |
|---|---|---|
| `0` | `CERT_NONE` | Client certificate not requested (default) |
| `1` | `CERT_OPTIONAL` | Client certificate requested but not required |
| `2` | `CERT_REQUIRED` | Client certificate required and verified |

### Client configuration

```bash
# Generate a client certificate signed by your CA
openssl req -newkey rsa:2048 -keyout client.key -out client.csr -nodes \
    -subj "/CN=my-client"
openssl x509 -req -in client.csr -CA client-ca.crt -CAkey client-ca.key \
    -CAcreateserial -out client.crt -days 365

# Make a request with the client certificate
curl https://your-server:8443/v1/completions \
    --cert client.crt \
    --key client.key \
    --cacert /etc/ssl/certs/vllm.crt \
    -H "Content-Type: application/json" \
    -d '{"model": "mistralai/Mistral-7B-Instruct-v0.3", "prompt": "Hello"}'
```

---

## Cipher suite configuration

Restrict the cipher suites used for TLS 1.2 and below:

```bash
vllm serve mistralai/Mistral-7B-Instruct-v0.3 \
    --ssl-keyfile key.pem \
    --ssl-certfile cert.pem \
    --ssl-ciphers "ECDHE-RSA-AES256-GCM-SHA384:ECDHE-RSA-CHACHA20-POLY1305:ECDHE-RSA-AES128-GCM-SHA256" \
    --host 0.0.0.0 \
    --port 8443
```

!!! note
    `--ssl-ciphers` applies to TLS 1.2 and below only. TLS 1.3 cipher suites
    are not configurable via this argument — they are managed by the OpenSSL
    library.

List available cipher suites on your system:

```bash
openssl ciphers -v 'HIGH:!aNULL:!MD5' | awk '{print $1}' | sort
```

---

## Docker deployment with TLS

Mount your certificates into the container:

```bash
docker run --runtime nvidia --gpus all \
    -v /etc/ssl/vllm:/etc/ssl/vllm:ro \
    -p 8443:8443 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    --ssl-keyfile /etc/ssl/vllm/server.key \
    --ssl-certfile /etc/ssl/vllm/server.crt \
    --host 0.0.0.0 \
    --port 8443
```

---

## Kubernetes deployment with TLS

### Store certificates in a Secret

```bash
kubectl create secret tls vllm-tls \
    --cert=/etc/ssl/certs/vllm.crt \
    --key=/etc/ssl/private/vllm.key \
    --namespace ns-vllm
```

### Mount the Secret in the Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-server
  namespace: ns-vllm
spec:
  template:
    spec:
      containers:
        - name: vllm
          image: vllm/vllm-openai:latest
          command:
            - vllm
            - serve
            - mistralai/Mistral-7B-Instruct-v0.3
            - --ssl-keyfile
            - /etc/ssl/vllm/tls.key
            - --ssl-certfile
            - /etc/ssl/vllm/tls.crt
            - --enable-ssl-refresh
            - --host
            - "0.0.0.0"
            - --port
            - "8443"
          ports:
            - containerPort: 8443
          volumeMounts:
            - name: tls-certs
              mountPath: /etc/ssl/vllm
              readOnly: true
      volumes:
        - name: tls-certs
          secret:
            secretName: vllm-tls
```

### Using cert-manager for automatic certificate management

[cert-manager](https://cert-manager.io/) automates certificate issuance and
renewal in Kubernetes. With `--enable-ssl-refresh`, vLLM automatically picks
up renewed certificates without a pod restart.

```yaml
apiVersion: cert-manager.io/v1
kind: Certificate
metadata:
  name: vllm-cert
  namespace: ns-vllm
spec:
  secretName: vllm-tls
  issuerRef:
    name: letsencrypt-prod
    kind: ClusterIssuer
  dnsNames:
    - api.example.com
```

---

## Security recommendations

| Recommendation | Details |
|---|---|
| Use TLS 1.2 or higher | Disable SSLv3, TLS 1.0, and TLS 1.1 |
| Prefer ECDHE cipher suites | Provides forward secrecy |
| Use certificates from a trusted CA | Avoid self-signed certs in production |
| Enable `--enable-ssl-refresh` | Avoid downtime during certificate renewal |
| Use mTLS for internal services | Authenticate clients as well as the server |
| Rotate certificates before expiry | Set up automated renewal (cert-manager, Certbot) |
| Store private keys in Secrets | Never embed keys in container images or ConfigMaps |

---

## Troubleshooting

### `SSL: CERTIFICATE_VERIFY_FAILED`

The client cannot verify the server certificate. Either:

- The certificate is self-signed — use `-k` with `curl` for testing, or add
  the CA to the client's trust store.
- The certificate hostname does not match — ensure the `CN` or `SAN` matches
  the hostname you are connecting to.

### `SSL: NO_SHARED_CIPHER`

The client and server have no cipher suites in common. Check the
`--ssl-ciphers` value and ensure it includes at least one cipher supported by
your client.

### Certificate not reloading

Ensure `--enable-ssl-refresh` is set and that the `watchfiles` package is
installed. Check the server logs for `SSLCertRefresher monitors files:` to
confirm the watcher started.

---

## Next steps

- [Load balancing](load_balancing.md) — TLS termination at the load balancer
- [Docker deployment](docker.md) — mount certificates into Docker containers
- [Kubernetes deployment](kubernetes.md) — manage certificates with cert-manager
- [Production checklist](production_checklist.md) — security hardening
