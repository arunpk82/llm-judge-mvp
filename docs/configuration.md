# Platform Configuration

The platform reads configuration from environment variables. Configuration
is validated **once** at `PlatformRunner` construction by
`llm_judge.control_plane.configuration.validate_configuration`. Misconfigurations
fail closed at startup rather than mid-execution.

## Environment variables

### `LLM_JUDGE_MODE`

Controls which configuration values are required.

| Value         | Behavior                                                                                               |
| ------------- | ------------------------------------------------------------------------------------------------------ |
| _unset_ / `"" `| Treated as `development` (default).                                                                    |
| `development` | Permissive: missing `LLM_JUDGE_CONTROL_PLANE_HMAC_KEY` is allowed and emits a one-shot startup warning. |
| `production`  | Strict: missing `LLM_JUDGE_CONTROL_PLANE_HMAC_KEY` raises `ConfigurationError`; platform refuses to start. |

Any other non-empty value (e.g. `staging`) raises `ConfigurationError` at startup
— typos do not silently degrade to a default.

### `LLM_JUDGE_CONTROL_PLANE_HMAC_KEY`

The HMAC-SHA256 key used to sign provenance envelopes. In `production` mode
this variable is required; in `development` mode it is optional and the
platform falls back to a publicly-knowable default key (`dev-key-not-for-prod`)
that is **not safe for any environment outside local development**.

## Modes in practice

### Local development

No configuration required. The platform starts, signs envelopes with the
default development key, and prints one warning at startup naming the
environment variable to set:

```
control_plane.config.default_hmac_key_in_use mode=development env_var=LLM_JUDGE_CONTROL_PLANE_HMAC_KEY
```

### Production deployments

Both variables must be set before the platform starts:

```
export LLM_JUDGE_MODE=production
export LLM_JUDGE_CONTROL_PLANE_HMAC_KEY=<your-secret-key>
```

If `LLM_JUDGE_CONTROL_PLANE_HMAC_KEY` is unset, empty, or whitespace-only,
`PlatformRunner()` raises `ConfigurationError` and the process exits before
any capability is invoked.

The HMAC key is also re-validated inside `_resolve_hmac_key` as
defense-in-depth: any envelope construction that bypasses
`PlatformRunner.__init__` (test isolation, partial mocks, future
direct-construction call sites) still refuses to sign with the development
default in production mode.

## Future extensions

`validate_configuration()` is the platform's startup-time configuration
surface. Subsequent packets absorb additional checks here:

- Layer vocabulary alignment between `wrappers.DEFAULT_LAYERS` and
  `eval/run.py` argparse choices (CP-F4, target packet L1-Pkt-2).
- Artifact root writability (`runs_root`, `transient_root`).
- Governance preflight reachability (`rubrics/registry.yaml`).
