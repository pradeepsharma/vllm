---
description: >
  Load balancing strategies for vLLM — Nginx round-robin and least-connections,
  elastic endpoint scaling, sleep mode, and Kubernetes-native approaches.
---

# Load balancing

Running multiple vLLM instances behind a load balancer increases throughput,
provides fault tolerance, and enables zero-downtime deployments. This page
covers Nginx-based load balancing, vLLM's built-in elastic endpoint scaling
API, sleep mode for resource management, and Kubernetes-native approaches.

---

## Overview

vLLM exposes a standard HTTP API, so any HTTP load balancer works. The key
considerations for LLM serving are:

- **Long-lived connections** — LLM inference requests can take seconds to
  minutes. Configure generous timeouts.
- **Streaming responses** — Server-Sent Events (SSE) require the load balancer
  to support chunked transfer encoding and not buffer responses.
- **Session affinity** — Prefix caching benefits from routing requests with
  the same prompt prefix to the same backend instance.
- **Health checking** — Use the `/health` endpoint to detect unhealthy
  instances before routing traffic to them.

---

## Nginx load balancing

Nginx is a lightweight, high-performance option for load balancing multiple
vLLM instances on a single host or across a Docker network.

### Docker Compose setup

The following example runs two vLLM instances behind an Nginx load balancer
using Docker.

#### Step 1: Build the vLLM image

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
docker build -f docker/Dockerfile --target vllm-openai -t vllm:local .
```

#### Step 2: Create the Nginx configuration

Create `nginx_conf/nginx.conf`:

```nginx
upstream vllm_backend {
    least_conn;
    server vllm0:8000 max_fails=3 fail_timeout=30s;
    server vllm1:8000 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;

    # Long timeouts for LLM inference
    proxy_read_timeout 3600s;
    proxy_send_timeout 3600s;
    proxy_connect_timeout 10s;

    location / {
        proxy_pass http://vllm_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Required for streaming (SSE) responses
        proxy_buffering off;
        proxy_cache off;
        proxy_http_version 1.1;
        chunked_transfer_encoding on;
    }

    location /health {
        proxy_pass http://vllm_backend;
        proxy_read_timeout 5s;
    }
}
```

#### Step 3: Create the Nginx Dockerfile

Create `Dockerfile.nginx`:

```dockerfile
FROM nginx:latest
RUN rm /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

Build it:

```bash
docker build -f Dockerfile.nginx -t nginx-lb .
```

#### Step 4: Create a Docker network

```bash
docker network create vllm_nginx
```

#### Step 5: Launch vLLM instances

```bash
hf_cache_dir=~/.cache/huggingface/

docker run -itd \
    --ipc host \
    --network vllm_nginx \
    --gpus device=0 \
    --shm-size=10g \
    -v $hf_cache_dir:/root/.cache/huggingface/ \
    -p 8081:8000 \
    --name vllm0 vllm:local \
    --model mistralai/Mistral-7B-Instruct-v0.3

docker run -itd \
    --ipc host \
    --network vllm_nginx \
    --gpus device=1 \
    --shm-size=10g \
    -v $hf_cache_dir:/root/.cache/huggingface/ \
    -p 8082:8000 \
    --name vllm1 vllm:local \
    --model mistralai/Mistral-7B-Instruct-v0.3
```

Wait for both instances to be ready:

```bash
docker logs vllm0 | grep Uvicorn
docker logs vllm1 | grep Uvicorn
# INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

#### Step 6: Launch Nginx

```bash
docker run -itd \
    -p 8000:80 \
    --network vllm_nginx \
    -v ./nginx_conf/:/etc/nginx/conf.d/ \
    --name nginx-lb nginx-lb:latest
```

#### Step 7: Test the load balancer

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "mistralai/Mistral-7B-Instruct-v0.3",
        "prompt": "San Francisco is a",
        "max_tokens": 16
    }'
```

---

## Load balancing algorithms

### Round-robin (default)

Distributes requests evenly across all backends in sequence:

```nginx
upstream vllm_backend {
    server vllm0:8000;
    server vllm1:8000;
}
```

### Least connections

Routes each new request to the backend with the fewest active connections.
This is the recommended algorithm for LLM serving because request durations
vary significantly:

```nginx
upstream vllm_backend {
    least_conn;
    server vllm0:8000;
    server vllm1:8000;
}
```

### IP hash (session affinity)

Routes requests from the same client IP to the same backend. Useful when
prefix caching is enabled and you want to maximise cache hit rates:

```nginx
upstream vllm_backend {
    ip_hash;
    server vllm0:8000;
    server vllm1:8000;
}
```

!!! note
    IP hash affinity is a coarse approximation. For fine-grained prefix-aware
    routing, consider the
    [vllm-project/production-stack](integrations/production-stack.md) or
    [llm-d](integrations/llm-d.md) integrations.

---

## TLS termination at the load balancer

Terminate TLS at Nginx and forward plain HTTP to vLLM backends. This is the
recommended approach for production because it centralises certificate
management:

```nginx
upstream vllm_backend {
    least_conn;
    server vllm0:8000;
    server vllm1:8000;
}

server {
    listen 443 ssl;
    server_name api.example.com;

    ssl_certificate     /etc/ssl/certs/api.example.com.crt;
    ssl_certificate_key /etc/ssl/private/api.example.com.key;
    ssl_protocols       TLSv1.2 TLSv1.3;
    ssl_ciphers         HIGH:!aNULL:!MD5;

    proxy_read_timeout 3600s;
    proxy_send_timeout 3600s;

    location / {
        proxy_pass http://vllm_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto https;
        proxy_buffering off;
    }
}

# Redirect HTTP to HTTPS
server {
    listen 80;
    server_name api.example.com;
    return 301 https://$host$request_uri;
}
```

---

## Elastic endpoint scaling

vLLM includes a built-in API for dynamically scaling the number of data
parallel workers without restarting the server. This is implemented in
`vllm/entrypoints/serve/elastic_ep/`.

### How it works

The elastic endpoint API exposes two endpoints:

- `POST /scale_elastic_ep` — triggers a scale operation to a new data parallel
  size.
- `POST /is_scaling_elastic_ep` — returns the current scaling state.

During a scale operation, a `ScalingMiddleware` intercepts all incoming
requests and returns `503 Service Unavailable` until scaling completes. This
prevents new requests from being routed to an instance that is mid-scale.

### Scaling to a new data parallel size

```bash
curl -X POST http://localhost:8000/scale_elastic_ep \
    -H "Content-Type: application/json" \
    -d '{
        "new_data_parallel_size": 4,
        "drain_timeout": 120
    }'
```

**Request parameters:**

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `new_data_parallel_size` | `int` | Yes | — | Target number of data parallel workers (must be > 0) |
| `drain_timeout` | `int` | No | `120` | Seconds to wait for in-flight requests to complete before scaling |

**Response:**

```json
{
    "message": "Scaled to 4 data parallel engines"
}
```

**Error responses:**

| Status | Condition |
|---|---|
| `400` | Missing or invalid `new_data_parallel_size` |
| `408` | In-flight requests did not drain within `drain_timeout` seconds |
| `500` | Internal scaling error |

### Checking scaling state

```bash
curl -X POST http://localhost:8000/is_scaling_elastic_ep
```

```json
{
    "is_scaling_elastic_ep": false
}
```

### Load balancer integration

When integrating elastic scaling with a load balancer:

1. Call `POST /is_scaling_elastic_ep` as a health check. Return `503` to the
   load balancer when `is_scaling_elastic_ep` is `true`.
1. The `ScalingMiddleware` already returns `503` for all requests during
   scaling, so the load balancer's health check will naturally detect the
   unavailable state.

```nginx
upstream vllm_backend {
    least_conn;
    server vllm0:8000;
    server vllm1:8000;
}

server {
    listen 80;

    location / {
        proxy_pass http://vllm_backend;
        proxy_next_upstream error timeout http_503;
        proxy_buffering off;
        proxy_read_timeout 3600s;
    }
}
```

The `proxy_next_upstream http_503` directive tells Nginx to retry the request
on the next backend when a `503` is received, transparently routing around an
instance that is currently scaling.

---

## Sleep mode

vLLM supports a sleep mode that frees GPU memory when the server is idle. This
is useful in multi-tenant environments where GPU resources are shared.

!!! note
    Sleep mode endpoints are only available when the server is started with
    `VLLM_SERVER_DEV_MODE=1`. They are intended for development and
    infrastructure tooling, not for direct use by API clients.

### Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/sleep` | `POST` | Put the engine to sleep and free GPU memory |
| `/wake_up` | `POST` | Wake the engine and restore GPU memory |
| `/is_sleeping` | `GET` | Check whether the engine is currently sleeping |

### Putting the engine to sleep

```bash
# Sleep level 1 (default) — abort in-flight requests
curl -X POST "http://localhost:8000/sleep?level=1&mode=abort"

# Sleep level 1 — wait for in-flight requests to complete
curl -X POST "http://localhost:8000/sleep?level=1&mode=wait"
```

**Query parameters:**

| Parameter | Default | Description |
|---|---|---|
| `level` | `1` | Sleep depth level |
| `mode` | `abort` | How to handle in-flight requests: `abort` or `wait` |

### Waking the engine

```bash
# Wake all tags
curl -X POST "http://localhost:8000/wake_up"

# Wake specific tags
curl -X POST "http://localhost:8000/wake_up?tags=model-a&tags=model-b"
```

### Checking sleep state

```bash
curl http://localhost:8000/is_sleeping
```

```json
{
    "is_sleeping": false
}
```

### Load balancer integration

Use `/is_sleeping` as a health check to prevent the load balancer from routing
requests to a sleeping instance:

```nginx
upstream vllm_backend {
    server vllm0:8000;
    server vllm1:8000;
}

server {
    listen 80;

    location / {
        proxy_pass http://vllm_backend;
        proxy_next_upstream error timeout http_503;
        proxy_buffering off;
        proxy_read_timeout 3600s;
    }
}
```

Configure an external health check script that calls `/is_sleeping` and marks
the backend as down when the engine is sleeping.

---

## Kubernetes load balancing

In Kubernetes, the `Service` resource provides built-in load balancing across
pod replicas using `kube-proxy` (iptables or IPVS).

### ClusterIP service (internal)

The Helm chart creates a `ClusterIP` service by default:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: vllm-service
  namespace: ns-vllm
spec:
  type: ClusterIP
  ports:
    - port: 80
      targetPort: 8000
      protocol: TCP
  selector:
    app: vllm
```

### LoadBalancer service (external)

To expose vLLM externally via a cloud load balancer:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: vllm-service
  namespace: ns-vllm
spec:
  type: LoadBalancer
  ports:
    - port: 443
      targetPort: 8000
      protocol: TCP
  selector:
    app: vllm
```

### Ingress with Nginx Ingress Controller

For path-based routing and TLS termination:

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: vllm-ingress
  namespace: ns-vllm
  annotations:
    nginx.ingress.kubernetes.io/proxy-read-timeout: "3600"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "3600"
    nginx.ingress.kubernetes.io/proxy-buffering: "off"
spec:
  tls:
    - hosts:
        - api.example.com
      secretName: vllm-tls
  rules:
    - host: api.example.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: vllm-service
                port:
                  number: 80
```

---

## Timeout configuration

LLM inference requests can be long-running. Configure timeouts generously:

| Component | Setting | Recommended value |
|---|---|---|
| Nginx `proxy_read_timeout` | Time to wait for a response from the backend | `3600s` |
| Nginx `proxy_send_timeout` | Time to wait between writes to the client | `3600s` |
| Nginx `proxy_connect_timeout` | Time to establish a connection to the backend | `10s` |
| Kubernetes Ingress `proxy-read-timeout` | Same as Nginx `proxy_read_timeout` | `3600` |
| vLLM `VLLM_HTTP_TIMEOUT_KEEP_ALIVE` | Keep-alive timeout for idle connections | `5` (default) |

---

## Health check endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | `GET` | Returns `200 OK` when the server is ready |
| `/is_sleeping` | `GET` | Returns sleeping state (dev mode only) |
| `/is_scaling_elastic_ep` | `POST` | Returns scaling state |

Use `/health` as the primary health check for all load balancers.

---

## Next steps

- [Docker deployment](docker.md) — run multiple vLLM instances with Docker
- [Kubernetes deployment](kubernetes.md) — Kubernetes Services and Ingress
- [SSL/TLS setup](ssl_tls.md) — TLS termination at vLLM or the load balancer
- [Production checklist](production_checklist.md) — production hardening
- [Nginx guide](nginx.md) — detailed Nginx configuration reference
