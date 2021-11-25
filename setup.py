import setuptools

setuptools.setup(
    name="dolfin_mech",
    version="2021.10.21",
    author="Martin Genet",
    author_email="martin.genet@polytechnique.edu",
    description=open("README.md", "r").readlines()[1][:-1],
    long_description = open("README.md", "r").read(),
    long_description_content_type="text/markdown",
    url="https://gitlab.inria.fr/mgenet/dolfin_mech",
    packages=["dolfin_mech"],
    license="GPLv3",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    install_requires=["numpy", "myPythonLibrary"],
)

# python -m keyring set https://upload.pypi.org/legacy/ martin.genet

# python setup.py sdist bdist_wheel
# python -m twine upload dist/*