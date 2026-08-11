# Legacy Converter Service

The converter runner installs the latest published Jittor package and exposes a
Flask service with caller-supplied TLS certificates. It is not a hardened or
reproducible production deployment.

It refuses to start without both an explicit `--run` acknowledgement and a TLS
directory:

```bash
CONVERTER_TLS_DIR=/etc/letsencrypt \
  tools/services/legacy/converter_server.sh --run
```

Review the image, certificate paths, public bind address, package version, and
resource limits before each use.
