import whitebox_workflows

# Activate your key
whitebox_workflows.activate_license(
key="889d88d3c7cacfcbd3ccc9cad3c8d3ced3cecacfcbc7d3cbcac9cdcdd3cec9c8cececec8c6cdcfcac7c9",
firstname="Abdiel",
lastname="Fernandez",
email="abdielfer@gmail.com",
agree_to_license_terms=True )

# Test your license by setting up the WbW environment
wbe = whitebox_workflows.WbEnvironment()