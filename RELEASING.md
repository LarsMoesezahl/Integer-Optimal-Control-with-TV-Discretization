# Releasing

## Create a new release tag

```bash
make release-tag
```

or manually:

```bash
git tag -a v0.2.0 -m "Release v0.2.0"
git push origin v0.2.0
```

## Suggested GitHub release notes

- reference `CHANGELOG.md`
- include environment/setup notes (`requirements.txt` / `environment.yml`)
- attach representative result plots from `results/`
