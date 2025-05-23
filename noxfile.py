import nox

@nox.session(python=["3.10"])
@nox.parametrize("versions", ["torch==1.12.1 numpy>=1.25,<1.26", "torch==2.7 numpy>=2.0"])
def tests(session, versions):
    session.install(".[test]", *versions.split(" "))
    session.run("pytest")
