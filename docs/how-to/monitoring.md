---
title: Monitor with Prometheus
description: Set up comprehensive training monitoring
---

# Monitor with Prometheus

Set up metrics export and dashboards.

## Enable Metrics

```yaml
logging:
  prometheus:
    enabled: true
    port: 9090
```

## Available Metrics

| Metric | Type | Description |
|:-------|:-----|:------------|
| `flux_training_loss` | Gauge | Current loss |
| `flux_staleness` | Gauge | Current staleness |
| `flux_async_ratio` | Gauge | Async ratio |
| `flux_throughput` | Gauge | Samples/second |
| `flux_gpu_utilization` | Gauge | GPU % |

## Prometheus Config

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'flux'
    static_configs:
      - targets: ['localhost:9090']
```

## Grafana Dashboard

Import dashboard JSON:

```json
{
  "panels": [
    {"title": "Loss", "targets": [{"expr": "flux_training_loss"}]},
    {"title": "Staleness", "targets": [{"expr": "flux_staleness"}]},
    {"title": "Throughput", "targets": [{"expr": "flux_throughput"}]}
  ]
}
```

## Alerts

```yaml
# alert.rules.yml
groups:
  - name: flux
    rules:
      - alert: HighStaleness
        expr: flux_staleness > 0.4
        for: 5m
        labels:
          severity: warning
```

## See Also

- [Production Deployment](../tutorials/production.md)
