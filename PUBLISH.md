# Publishing hypertrack to PyPI

This guide explains how to publish the `hypertrack` package to PyPI.

## Prerequisites

1. **PyPI Account**: Create an account at https://pypi.org/account/register/
2. **API Token**: Generate an API token at https://pypi.org/manage/account/token/
3. **Build Tools**: Install build and upload tools:

```bash
pip install build twine
```

## Step 1: Update Package Metadata

Before publishing, update the following files with your information:

1. **setup.py**: Update `author`, `author_email`, and `url`
2. **pyproject.toml**: Update `authors` and `project.urls`
3. **README.md**: Update any GitHub URLs if needed

## Step 2: Build the Package

Build both source distribution (sdist) and wheel:

```bash
python -m build
```

This creates:
- `dist/hypertrack-0.1.0.tar.gz` (source distribution)
- `dist/hypertrack-0.1.0-py3-none-any.whl` (wheel)

## Step 3: Check the Package

Before uploading, check the package:

```bash
# Check the built package
twine check dist/*
```

## Step 4: Test on TestPyPI (Recommended)

First, test on TestPyPI to ensure everything works:

```bash
# Upload to TestPyPI
twine upload --repository testpypi dist/*

# Test installation from TestPyPI
pip install --index-url https://test.pypi.org/simple/ hypertrack
```

## Step 5: Upload to PyPI

Once tested, upload to the real PyPI:

```bash
twine upload dist/*
```

You'll be prompted for your PyPI credentials. Use your API token:
- Username: `__token__`
- Password: Your API token (starts with `pypi-`)

Or set environment variables:

```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-your-token-here
twine upload dist/*
```

## Step 6: Verify Installation

After uploading, verify the package is available:

```bash
pip install hypertrack
```

## Versioning

When releasing a new version:

1. Update version in:
   - `setup.py` (version="X.Y.Z")
   - `pyproject.toml` (version = "X.Y.Z")
   - `hypertrack/__init__.py` (__version__ = "X.Y.Z")

2. Create a git tag:

```bash
git tag v0.1.0
git push origin v0.1.0
```

3. Build and upload the new version

## Troubleshooting

### "File already exists" error

This means the version already exists on PyPI. Increment the version number.

### Authentication errors

- Make sure you're using `__token__` as username
- Ensure your API token has the correct permissions
- Check that the token hasn't expired

### Build errors

- Ensure all required files are included (check MANIFEST.in)
- Verify setup.py and pyproject.toml are correct
- Check for syntax errors in Python files

## Additional Resources

- [PyPI Documentation](https://packaging.python.org/tutorials/packaging-projects/)
- [Twine Documentation](https://twine.readthedocs.io/)
- [Python Packaging Guide](https://packaging.python.org/)
