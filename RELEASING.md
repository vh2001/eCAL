# Releasing eCAL

## Cutting a new version

1. Bump `version` in `pyproject.toml`.
2. Open a PR with the version bump, targeting `main`.
3. Get it reviewed and merged.
4. On GitHub → **Releases → Draft a new release**:
   - Tag: `vX.Y.Z` — **must exactly match** the `pyproject.toml` version (e.g. `v0.1.1` for `0.1.1`), or the workflow fails on purpose.
   - Target: `main`
   - Fill in release notes, then **Publish release**.
5. This triggers `.github/workflows/publish.yml`: builds the sdist/wheel, verifies the tag matches `pyproject.toml`, then pauses waiting for approval.
6. Someone with access to the `pypi` environment approves the deployment (**Actions tab → the run → Review deployments**). Currently, either **vh2001** or **cfortuna** can approve.
7. Verify the new version is live at https://pypi.org/project/ecal-energy/.

Docs redeploy automatically to https://sensorlab.github.io/eCAL/ on every merge to `main` — no separate step needed.
